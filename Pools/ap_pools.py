from jqdata import *
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict
import numpy as np
import pandas as pd
import datetime
import unicodedata
import builtins
import math
from pca_analysis import analyze_pool_pca
from join_method import *

# -------------------- 配置区域 (Initial Configuration) --------------------
CONFIG = {
    # 基础过滤参数
    "min_liquidity": 50e6,       # 最小成交额 (5000万)
    "min_listing_days": 60,      # 最小上市天数
    "correlation_window": 120,      # 计算相关性的回看窗口 (天)
    "smoothing_steps": 1,        # 恢复原状：1表示不平滑，只计算当前窗口
    "smoothing_lag": 20,         # (此参数在steps=1时无效)
    
    # 类型过滤配置 (True=剔除, False=保留)
    "filter_bond_money": True,    # 剔除债券、货币
    "filter_qdii": False,         # 剔除跨境/QDII
    "filter_gold": False,         # 剔除黄金
    "black_list": ["510900.XSHG"],  # 黑名单
    
    # 聚类方法选择 ("hierarchical", "ap", "mst", "dbscan")
    "clustering_method": "ap",

    # Hierarchical聚类参数
    "cluster_corr_threshold": 0.85, # 聚类相关性阈值 (决定簇的数量/大小)
    
    # AP聚类参数
    "ap_damping": 0.8,           # 阻尼系数 (0.5-1.0)，防止震荡，越大越稳定
    "ap_preference": None,       # 偏好值 (None = 使用中位数)，值越大簇越多
    
    # MST聚类参数
    "mst_n_clusters": 10,        # MST聚类目标簇数量 (切断由强到弱的K-1条边)

    # DBSCAN聚类参数
    "dbscan_eps": 0.5,           # DBSCAN 邻域半径
    "dbscan_min_samples": 2,     # DBSCAN 最小样本数
    
    # 动量评分参数
    "score_window": 25,          # 动量评分的回看窗口 (天)
    "min_r2": 0.7,               # 最小 R 方
    "max_annualized_returns": 9.99, # 最大年化收益 (过滤异常值)
    
    # PCA 分析参数
    "pca_rolling_window": 60,    # 滚动 PCA 窗口 (天)
    "pca_overlay_etf": "510300.XSHG", # 叠加显示的基准 ETF (如沪深300)，为 None 则不显示
}

# -------------------- 工具函数 --------------------------
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

def _width(s: str) -> int:
    """返回字符串在终端里真实占的列数"""
    return builtins.sum(2 if unicodedata.east_asian_width(ch) in 'FW' else 1 for ch in s)

def pretty_print(df: pd.DataFrame, sep: str = '  ') -> None:
    """强制左对齐、屏幕宽度对齐、不换行打印 DataFrame"""
    str_df = df.astype(str)
    cols = list(df.columns)
    widths = {}
    for c in cols:
        w = max(_width(str(v)) for v in str_df[c])
        widths[c] = max(w, _width(c))
    
    header = sep.join(c.ljust(widths[c] - (_width(c) - len(c))) for c in cols)
    print(header)
    for _, row in str_df.iterrows():
        line = sep.join(
            row[c].ljust(widths[c] - (_width(row[c]) - len(row[c]))) for c in cols
        )
        print(line)

# -------------------- 1. 初始池筛选 --------------------------
def initial_etf_filter(target_date=None, verbose=True, config=CONFIG):
    """
    基础过滤：
    1. 获取所有ETF
    2. 剔除黑名单、非核心深市、特定前缀(债券/货币等)
    3. 剔除上市时间过短的
    """
    if verbose: print(f"Step 1: Initial Filtering (Date: {target_date})...")
    if target_date is None:
        target_date = datetime.date.today()
        
    df = get_all_securities(["etf"], date=target_date)
    
    # 白名单/黑名单
    blacklist = config['black_list']
    df = df[~df.index.isin(blacklist)]
    
    # 关键字剔除 (主要针对深市及名字中包含特定标识的ETF)
    exclude_keywords = []
    # if config["filter_bond_money"]: 
    #     exclude_keywords.extend(["债", "货币", "理财"])
    if config["filter_gold"]:       
        exclude_keywords.extend(["黄金", "上海金"])
    if config["filter_qdii"]:       
        exclude_keywords.extend(["QDII", "标普", "纳指", "道琼斯", "恒生", "H股", "日经", "德国", "法国", "英国", "美国", "海外"])
    
    if exclude_keywords:
        if verbose: print(f"  -> Excluding keys: {exclude_keywords}")
        # using regex to filter
        pattern = "|".join(exclude_keywords)
        df = df[~df['display_name'].str.contains(pattern, regex=True)]
    
    # 上市时间检查
    # Use target_date instead of yesterday
    check_date = target_date - datetime.timedelta(days=1)
    df = df[((check_date - df["start_date"]).dt.days >= config["min_listing_days"]) & (df["end_date"] > check_date)]

    # ----------------------------------------------------
    # 新增：价格过滤 (针对债券、货币)
    # 策略：如果最近1日均价 > 90，则认为是债券或货币基金
    # ----------------------------------------------------
    if config["filter_bond_money"]:
        if verbose: print("  -> Checking for high-priced ETFs (Bond/Money > 90)...")
        
        try:
            # avg_prices_df = history(1, "1d", "avg", df.index.tolist(), df=True)
            avg_prices_df = get_history_data(df.index.tolist(), target_date, 1, "avg")
            # Usually get_history_data returns index=Date, columns=Securities.
            
            if not avg_prices_df.empty:
                # avg_prices_df: rows=date (1 row), cols=securities
                last_prices = avg_prices_df.iloc[-1]
                
                # 找出价格 > 90 的代码
                high_price_etfs = last_prices[last_prices > 90].index.tolist()
                if high_price_etfs:
                    if verbose: print(f"  -> Filtering out {len(high_price_etfs)} ETFs with price > 90 (likely Bond/Money):")
                    # print(f"     Examples: {high_price_etfs[:5]}")
                    df = df[~df.index.isin(high_price_etfs)]
        except Exception as e:
            print(f"  [Warning] Failed to filter by price: {e}")

    initial_list = df.index.tolist()
    if verbose: print(f"  -> Found {len(initial_list)} candidate ETFs.")
    return initial_list

# -------------------- 3. 动量评分 --------------------------
def get_best_etf(etf_list, config=CONFIG):
    """
    计算动量得分并排序
    """
    print("Step 3: Momentum Scoring & Ranking...")
    data = pd.DataFrame(index=etf_list, columns=["code", "name", "annualized_returns", "r2", "score"])
    
    for etf in etf_list:
        # 获取数据
        df = get_price(etf, end_date=today, frequency="daily", fields=["close"], count=config["score_window"], panel=False)
        if df.empty:
            continue
            
        prices = df["close"].values
        
        # 线性回归 (log price)
        y = np.log(prices)
        x = np.arange(len(y))
        
        # 加权 (近期权重更高)
        weights = np.linspace(1, 2, len(y))
        
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        
        # 年化收益
        ann_ret = round(math.exp(slope * 250) - 1, 2)
        
        # R2
        y_pred = slope * x + intercept
        ss_res = np.sum(weights * (y - y_pred) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        r2 = round(1 - ss_res / ss_tot if ss_tot else 0, 2)
        
        # 得分
        score = round(ann_ret * r2, 2)
        
        data.loc[etf, "code"] = etf
        try:
            data.loc[etf, "name"] = get_security_info(etf).display_name
        except:
            data.loc[etf, "name"] = "Unknown"
        data.loc[etf, "annualized_returns"] = ann_ret
        data.loc[etf, "r2"] = r2
        data.loc[etf, "score"] = score
        
        # 简单止损过滤：最近3天跌幅超过5%则归零 (参考原逻辑)
        if len(prices) >= 4:
             if min(prices[-1]/prices[-2], prices[-2]/prices[-3], prices[-3]/prices[-4]) < 0.95:
                 data.loc[etf, "score"] = 0

    # 排序与过滤
    filtered = data[
        (data["score"] > 0) & 
        (data["annualized_returns"] < config["max_annualized_returns"]) & 
        (data["r2"] > config["min_r2"])
    ].sort_values(by="score", ascending=False)
    
    return filtered
    return filtered

# -------------------- 4. 多轮聚合 (Multi-Round Aggregation) --------------------------
def multi_round_aggregation(freq="M", periods=6, end_date=None, verbose=False, config=CONFIG):
    """
    Perform aggregation for multiple dates looking back from a time series.
    freq: 'M' (Month End), 'W' (Weekly), etc.
    periods: Number of periods to look back.
    end_date: Setup end date.
    
    Output: Table of selected ETFs for each date.
    """
    if end_date is None:
        end_date = datetime.date.today()
    
    # Generate date range
    # pandas date_range is powerful
    # We want points *ending* at end_date, going back.
    # e.g. end_date='2025-10-30', periodic check points.
    
    dates = pd.date_range(end=end_date, periods=periods, freq=freq).tolist()
    # Convert to date objects
    dates = [d.date() for d in dates]
    
    print("=" * 60)
    print(f"Multi-Round Aggregation ({len(dates)} rounds)")
    print(f"Dates: {dates}")
    print(f"Method: {config['clustering_method']}")
    print("=" * 60)
    
    results = []
    
    for d in dates:
        print(f"\nProcessing Date: {d}")
        try:
            # 1. Initial Filter
            candidates = initial_etf_filter(target_date=d, verbose=verbose)
            if not candidates:
                print("  -> No candidates found.")
                results.append({"date": d, "count": 0, "etfs": ""})
                continue
                
            # 2. Clustering
            method = config["clustering_method"]
            if method == "ap":
                final_pool = ap_clustering_filter(candidates, config, target_date=d, verbose=verbose)
            elif method == "mst":
                final_pool = mst_clustering_filter(candidates, config, target_date=d, verbose=verbose)
            elif method == "dbscan":
                final_pool = dbscan_clustering_filter(candidates, config, target_date=d, verbose=verbose)
            else:
                final_pool = hierarchical_clustering_filter(candidates, config, target_date=d, verbose=verbose)
            
            # Record result
            # Convert list to string
            etf_str = ",".join([get_security_info(c).display_name for c in final_pool])
            results.append({"date": d, "count": len(final_pool), "etfs": etf_str, "codes": final_pool})
            
        except Exception as e:
            print(f"  [Error] Processing {d} failed: {e}")
            results.append({"date": d, "count": 0, "etfs": "ERROR"})

    # Output Table
    print("\n" + "=" * 60)
    print("Multi-Round Aggregation Results")
    print("=" * 60)
    
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        pretty_print(res_df[["date", "count", "etfs"]])
        
    return res_df

# -------------------- 5. 稳定性分析工具 --------------------------
def analyze_cluster_stability(base_date=None, lag_days=1, verbose=True, config=CONFIG):
    """
    研究两个相似时间点（base_date vs base_date - lag_days）的聚类结果偏差。
    用于量化观察聚类算法的时序不稳定性。
    """
    if base_date is None:
        base_date = datetime.date.today()
        
    prev_date = base_date - datetime.timedelta(days=lag_days)
    
    print("=" * 60)
    print(f"Stability Analysis: {base_date} vs {prev_date} (Lag: {lag_days} days)")
    print(f"Algorithm: {config['clustering_method']}")
    print("=" * 60)
    
    # Run 1: Base Date
    print(f"\n[Run 1] Processing {base_date}...")
    pool1 = []
    details1 = {}
    try:
        c1 = initial_etf_filter(target_date=base_date, verbose=False)
        if config["clustering_method"] == "ap":
            pool1, details1 = ap_clustering_filter(c1, config, target_date=base_date, verbose=False, return_details=True)
        elif config["clustering_method"] == "mst":
            pool1, details1 = mst_clustering_filter(c1, config, target_date=base_date, verbose=False, return_details=True)
        elif config["clustering_method"] == "dbscan":
            pool1, details1 = dbscan_clustering_filter(c1, config, target_date=base_date, verbose=False, return_details=True)
        else:
            pool1, details1 = hierarchical_clustering_filter(c1, config, target_date=base_date, verbose=False, return_details=True)
    except Exception as e:
        print(f"  Error in Run 1: {e}")

    # Run 2: Previous Date
    print(f"[Run 2] Processing {prev_date}...")
    pool2 = []
    details2 = {}
    try:
        c2 = initial_etf_filter(target_date=prev_date, verbose=False)
        if config["clustering_method"] == "ap":
            pool2, details2 = ap_clustering_filter(c2, config, target_date=prev_date, verbose=False, return_details=True)
        elif config["clustering_method"] == "mst":
            pool2, details2 = mst_clustering_filter(c2, config, target_date=prev_date, verbose=False, return_details=True)
        elif config["clustering_method"] == "dbscan":
            pool2, details2 = dbscan_clustering_filter(c2, config, target_date=prev_date, verbose=False, return_details=True)
        else:
            pool2, details2 = hierarchical_clustering_filter(c2, config, target_date=prev_date, verbose=False, return_details=True)
    except Exception as e:
        print(f"  Error in Run 2: {e}")
        
    # Analysis
    set1 = set(pool1)
    set2 = set(pool2)
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # 1. Selection Stability (Jaccard)
    stability_score = len(intersection) / len(union) if union else 0
    
    # 2. Structural Stability (ARI)
    # Find common ETFs in both clustering runs (candidates that survived filtering in both)
    # Note: details keys are the 'final_etf_list' used in clustering, which includes all valid candidates before selection.
    
    common_candidates = set(details1.keys()) & set(details2.keys())
    ari_score = 0.0
    
    if common_candidates:
        labels_true = [details1[etf] for etf in common_candidates]
        labels_pred = [details2[etf] for etf in common_candidates]
        ari_score = adjusted_rand_score(labels_true, labels_pred)
    
    print("\n" + "-"*30 + " Results " + "-"*30)
    print(f"Pool 1 Selected ({base_date}): {len(pool1)}")
    print(f"Pool 2 Selected ({prev_date}): {len(pool2)}")
    print(f"Selection Stability (Jaccard): {stability_score:.4f} (1.0 = Identical Selection)")
    
    print(f"\nCommon Candidates Analysis (Structure):")
    print(f"Common candidate ETFs count: {len(common_candidates)}")
    print(f"Structural Stability (ARI):    {ari_score:.4f} (1.0 = Identical Grouping)")
    print(f"  (ARI measures how similar the clustering partitions are for the same set of assets)")
    
    # Changes
    newly_added = set1 - set2
    dropped = set2 - set1
    
    print("\n" + "-"*30 + " Results " + "-"*30)
    print(f"Pool 1 Size ({base_date}): {len(pool1)}")
    print(f"Pool 2 Size ({prev_date}): {len(pool2)}")
    print(f"Intersection: {len(intersection)}")
    print(f"Stability Score (Jaccard): {stability_score:.4f} (1.0 = Identical)")
    
    print(f"\nChanged: {len(newly_added) + len(dropped)}")
    if newly_added:
        names = [f"{get_security_info(c).display_name}" for c in newly_added]
        print(f"  + Added ({len(newly_added)}): {', '.join(names)}")
    if dropped:
        names = [f"{get_security_info(c).display_name}" for c in dropped]
        print(f"  - Dropped ({len(dropped)}): {', '.join(names)}")
        
    return stability_score

# -------------------- 主程序 --------------------------
if __name__ == "__main__":
    
    # Mode Switch: "SINGLE", "MULTI", "STABILITY"
    MODE = "STABILITY" 
    
    if MODE == "STABILITY":
        analyze_cluster_stability(lag_days=1) # Comparative Analysis
        exit()
    
    if MODE == "MULTI":
        # Example: Last 6 month-ends
        multi_round_aggregation(freq="ME", periods=6, verbose=False)
        print("\n" + "-"*30 + " Single Round (Today) " + "-"*30)

    print("-" * 50)
    print(f"Starting {CONFIG['clustering_method'].upper()}-Based ETF Pool Generation")
    print("Configuration:", CONFIG)
    print("-" * 50)
    
    # 1. 初始筛选
    candidates = initial_etf_filter(target_date=today)
    
    # 2. 聚类选择
    if CONFIG["clustering_method"] == "ap":
        final_pool = ap_clustering_filter(candidates, CONFIG, target_date=today)
        method_name = "Affinity Propagation"
    elif CONFIG["clustering_method"] == "mst":
        final_pool = mst_clustering_filter(candidates, CONFIG, target_date=today)
        method_name = "Minimum Spanning Tree (MST)"
    elif CONFIG["clustering_method"] == "dbscan":
        final_pool = dbscan_clustering_filter(candidates, CONFIG, target_date=today)
        method_name = "DBSCAN Clustering"
    else:
        final_pool = hierarchical_clustering_filter(candidates, CONFIG, target_date=today)
        method_name = "Hierarchical Clustering"
        
    print(f"{method_name} Complete. Pool Size: {len(final_pool)}")
    
    # 2.5 PCA 分析
    analyze_pool_pca(final_pool, CONFIG, overlay_code=CONFIG.get("pca_overlay_etf"))
    
    # 3. 动量评分展示
    result_df = get_best_etf(final_pool)
    
    print("\nFinal Selected ETF Pool (Top Momentum):")
    pretty_print(result_df)
