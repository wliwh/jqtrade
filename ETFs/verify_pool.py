# 验证脚本：用于在 JQ 研究环境中测试动态池生成逻辑
# 路径：ETFs/verify_pool.py

from jqdata import *
import pandas as pd
from datetime import datetime, timedelta
import re
from sklearn.cluster import AffinityPropagation
from collections import defaultdict

# ============= 全局常量与模拟参数 =============
ETF_WHITELIST = ['518880.XSHG', '511380.XSHG', '511260.XSHG', '511090.XSHG']
A_SHARE_INDEX_REGEX = r'^(H\d{5}\.CSI|\d{6}\.(CSI|XSHG|XSHE))$'

POOL_MIN_LISTED_DAYS = 60
POOL_VOL_LOOKBACK = 20
POOL_MIN_AVG_MONEY = 2000*10000 # 2000万
POOL_FILTER_BOND_MONEY = True
POOL_FILTER_QDII = False
POOL_FILTER_A_SHARE = True  # 默认开启测试

# AP 聚类参数
POOL_USE_AP_CLUSTERING = False     # 是否使用 AP 聚类
AP_DAMPING = 0.5
AP_PREFERENCE = None
AP_CORR_WINDOW = 60

def test_pool_generation():
    print(f"正在测试动态池生成逻辑... (模拟日期: {datetime.now().strftime('%Y-%m-%d')})")
    current_date = datetime.now().date()
    
    # 1. 获取全市场ETF + 关键词过滤
    all_etfs = get_all_securities(types=['etf'])
    exclude_keywords = []
    if POOL_FILTER_BOND_MONEY: exclude_keywords.extend(["债", "货币", "理财"])
    if POOL_FILTER_QDII: exclude_keywords.extend(["QDII", "标普", "纳指", "道琼斯", "恒生", "H股", "日经", "德国", "法国", "英国", "美国", "海外"])
    if POOL_FILTER_A_SHARE: exclude_keywords.extend(["300", "500", "50", "800", "1000", "2000", "中证", "创业板", "科创", "沪深", "上证", "深证", "A股"])
    if exclude_keywords:
        pattern = "|".join(exclude_keywords)
        all_etfs = all_etfs[~all_etfs['display_name'].str.contains(pattern, regex=True)]
    print(f"Step 1: 关键词过滤后剩余 ETF 数量: {len(all_etfs)}")
    
    # 2. 过滤上市时间
    min_date = current_date - timedelta(days=POOL_MIN_LISTED_DAYS)
    candidates = all_etfs[all_etfs['start_date'] < min_date].index.tolist()
    print(f"Step 2: 上市时限过滤后数量: {len(candidates)}")
    if not candidates: return

    # 3. 流动性过滤
    prices = get_price(candidates, count=POOL_VOL_LOOKBACK, end_date=current_date, 
                      frequency='daily', fields=['money'], panel=False)
    if prices is None or prices.empty: return
    avg_money = prices.groupby('code')['money'].mean()
    liquid_etfs = avg_money[avg_money >= POOL_MIN_AVG_MONEY].index.tolist()
    
    # 3.1 价格过滤 (针对 100元左右的债基)
    if POOL_FILTER_BOND_MONEY and liquid_etfs:
        hist_avg = get_price(liquid_etfs, count=1, end_date=current_date, frequency='daily', fields=['avg'], panel=False)
        if hist_avg is not None and not hist_avg.empty:
            last_avg = hist_avg.groupby('code')['avg'].last()
            bond_etfs = last_avg[last_avg > 90].index.tolist()
            liquid_etfs = [e for e in liquid_etfs if e not in bond_etfs]
    print(f"Step 3: 流动性与价格过滤后数量: {len(liquid_etfs)}")

    # 4. 指数级精准过滤与初步去重 (Stage 1)
    print("\n[Stage 1] 执行指数级精准过滤与去重...")
    index_deduped_list = []
    try:
        q = query(finance.FUND_INVEST_TARGET).filter(finance.FUND_INVEST_TARGET.code.in_(liquid_etfs))
        df_target = finance.run_query(q)
        
        # 4a. A 股指数代码正则过滤
        if POOL_FILTER_A_SHARE and not df_target.empty:
            mask = df_target['traced_index_code'].str.match(A_SHARE_INDEX_REGEX, na=False)
            a_share_codes = df_target.loc[mask, 'code'].unique().tolist()
            if a_share_codes:
                print(f"正则匹配命中 {len(a_share_codes)} 只 A 股 ETF 并剔除")
                liquid_etfs = [c for c in liquid_etfs if c not in a_share_codes]
                df_target = df_target[~df_target['code'].isin(a_share_codes)]
        
        # 4b. 指数去重
        if not df_target.empty:
            df_target['start_date'] = pd.to_datetime(df_target['start_date'])
            df_target = df_target[df_target['start_date'].dt.date <= current_date]
            current_index_map = df_target.sort_values('start_date', ascending=False).drop_duplicates('code', keep='first')
            current_index_map['group_key'] = current_index_map['traced_index_code'].replace('', None).fillna(current_index_map['traced_index_name'])
            current_index_map['avg_money'] = current_index_map['code'].map(avg_money)
            
            valid_grouped = current_index_map[current_index_map['group_key'].notna()]
            no_index_etfs = [c for c in liquid_etfs if c not in valid_grouped['code'].tolist()]
            best_etfs = valid_grouped.sort_values('avg_money', ascending=False).drop_duplicates('group_key', keep='first')
            index_deduped_list = best_etfs['code'].tolist() + no_index_etfs
        else:
            index_deduped_list = liquid_etfs
    except Exception as e:
        print(f"指数处理异常: {e}")
        index_deduped_list = liquid_etfs
    # 4.5 白名单保底 (纳入第一阶段过滤后，进入聚类前)
    for wl in ETF_WHITELIST:
        if wl in liquid_etfs and wl not in index_deduped_list:
            index_deduped_list.append(wl)
    print(f"Stage 1 结束，指数去重后数量: {len(index_deduped_list)}")

    # 5. AP 聚类优化 (Stage 2 - 可选)
    final_pool = index_deduped_list
    if POOL_USE_AP_CLUSTERING:
        print("\n[Stage 2] 执行 AP 聚类优化...")
        try:
            prices_ap = get_price(index_deduped_list, count=AP_CORR_WINDOW+1, end_date=current_date, frequency='daily', fields=['close'], panel=False)
            if prices_ap is not None and not prices_ap.empty:
                close_pivot = prices_ap.pivot(index='time', columns='code', values='close').fillna(method='ffill').dropna(axis=1, how='any')
                returns = close_pivot.pct_change().dropna()
                if len(returns.columns) >= 2:
                    ap = AffinityPropagation(damping=AP_DAMPING, preference=AP_PREFERENCE, affinity='precomputed')
                    ap.fit(returns.corr())
                    cluster_dict = defaultdict(list)
                    for etf, label in zip(returns.columns, ap.labels_):
                        cluster_dict[label].append(etf)
                    
                    ap_excluded = [e for e in index_deduped_list if e not in returns.columns]
                    if ap_excluded:
                        print(f"数据缺失，未能参与聚类（强制保留）支数: {len(ap_excluded)}")
                    final_pool = [max(etfs, key=lambda x: avg_money.get(x, 0)) for etfs in cluster_dict.values()]
                    final_pool += ap_excluded
                    print(f"Stage 2 结束，聚类压缩后数量: {len(final_pool)}")
        except Exception as e:
            print(f"AP 聚类失败: {e}")

    print("\n" + "="*50)
    print(f"最终生成的动态池 (共 {len(final_pool)} 只):")
    print("="*50)
    results = []
    for c in final_pool:
        info = get_security_info(c)
        results.append({
            'code': c,
            'display_name': info.display_name,
            'avg_money': avg_money.get(c, 0)
        })
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        print(df_res.sort_values('avg_money', ascending=False).to_string(index=False))
    return df_res

if __name__ == "__main__":
    test_pool_generation()
