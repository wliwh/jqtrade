from jqdata import *
import logging
from jqfactor import *
import pandas as pd
import numpy as np
import datetime
import math
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from collections import defaultdict

logger = logging.getLogger(__name__)

def get_history_data(security_list, end_date, count, field="close"):
    """
    辅助函数：使用 get_price 获取截至特定日期的历史数据。
    处理严格的格式以返回 DataFrame (Index=Date, Columns=Securities)。
    """
    # 确保输入是列表
    if isinstance(security_list, str):
        security_list = [security_list]

    try:
        # 用户确认 get_price 带 panel=False 返回扁平 DataFrame
        df = get_price(list(security_list), end_date=end_date, count=count, 
                       frequency='daily', fields=[field], panel=False)
        
        if df.empty:
            return pd.DataFrame()

        # 1. 处理扁平 DataFrame (多个标的) 使用 Pivot
        if 'code' in df.columns:
            index_col = 'time' if 'time' in df.columns else 'date' if 'date' in df.columns else None
            
            if index_col:
                # Pivot to Index=Date, Columns=Code
                return df.pivot(index=index_col, columns='code', values=field)
            else:
                return df.pivot(columns='code', values=field)

        # 2. 处理单个标的 (Index = Date)
        if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)) or (len(df) > 0 and isinstance(df.index[0], (datetime.date, datetime.datetime, pd.Timestamp))):
            if len(security_list) == 1:
                security_code = security_list[0]
                if field in df.columns:
                     return df[[field]].rename(columns={field: security_code})
            
            return df[field] # Fallback

        # 3. 旧版 Panel 或 ndim=3
        if getattr(df, "ndim", 2) == 3:
             return df[field]
             
        # 4. MultiIndex (Date, Code) - 新版 JQData 有时会出现
        if isinstance(df.index, pd.MultiIndex):
            unstacked = df.unstack(level=1)
            if field in unstacked.columns:
                return unstacked[field]
            if field in unstacked.columns.get_level_values(0):
                 return unstacked.xs(field, axis=1, level=0, drop_level=True)
            return unstacked
        return df[field]

    except Exception as e:
        logger.error(f"  [错误] 获取历史数据失败: {e}")
        return pd.DataFrame()

# -------------------- 1.5 核心工具：平滑相关性计算 --------------------------
def get_smoothed_correlation(etf_list, config, target_date=None):
    """
    计算平滑后的相关性矩阵，以保证聚类结果的时序稳定性。
    逻辑：计算过去N个时间窗口的相关性矩阵平均值。
    """
    if target_date is None: target_date = datetime.date.today()
    
    # 1. 获取参数
    window = config["correlation_window"]
    steps = config.get("smoothing_steps", 1)  # 平滑步数，默认1(不平滑)
    lag = config.get("smoothing_lag", 20)     # 每次滞后的天数，默认20(约1个月)
    
    if steps > 1:
        logger.debug(f"  -> 计算平滑相关性 (步数={steps}, 滞后={lag}天)...")
        
    # 2. 计算所需总数据长度
    # 需要的数据长度 = 窗口 + (步数-1) * 滞后
    total_count = window + (steps - 1) * lag
    
    # 3. 获取数据
    price_data = get_history_data(etf_list, target_date, total_count, "close")
    price_data = price_data.fillna(method="ffill").dropna(axis=1, how="any")
    
    if price_data.empty:
        logger.debug("  -> 价格数据为空。")
        return None, []
        
    # 计算所有日期的收益率
    all_returns = price_data.pct_change().dropna()
    valid_etfs = all_returns.columns.tolist()
    
    if len(valid_etfs) < len(etf_list):
        logger.debug(f"  -> 因数据缺失丢弃了 {len(etf_list)-len(valid_etfs)} 个 ETF。")
        
    # 4. 滚动计算相关性矩阵并求和
    sum_corr = None
    count = 0
    
    # 数据长度: all_returns 的行数 (N)
    # 我们需要取切片: [end - window : end]
    n_samples = len(all_returns)
    
    for i in range(steps):
        # 计算切片索引
        # 第0步(current): end_idx = n_samples
        # 第1步(lag 1):   end_idx = n_samples - lag
        end_idx = n_samples - i * lag
        start_idx = end_idx - window
        
        # 边界检查
        if start_idx < 0:
            logger.warning(f"  -> 警告: 平滑步骤 {i+1} 数据不足。")
            break
            
        # 切片收益率
        slice_returns = all_returns.iloc[start_idx:end_idx]
        
        if len(slice_returns) < window * 0.9: # 简单数据量检查
             continue
             
        # 计算该窗口的相关性
        corr = slice_returns.corr(method="spearman")
        
        if sum_corr is None:
            sum_corr = corr
        else:
            sum_corr += corr
        count += 1
        
    if count == 0:
        return None, []
        
    # 5. 取平均
    avg_corr = sum_corr / count
    return avg_corr, valid_etfs

# -------------------- 1.6 核心工具：获取排序依据数据 --------------------------
def get_ranking_data(etf_list, target_date, method='money'):
    """
    获取用于筛选代表性ETF的排名数据
    method: 'money' (30日平均成交额), 'market_cap' (总市值)
    """
    if method == 'money':
        # 保持原逻辑：30日成交额均值
        return get_history_data(etf_list, target_date, 30, "money").mean()
        
    elif method == 'market_cap':
        # 获取截至target_date的总市值
        try:
            q = query(valuation.code, valuation.market_cap).filter(
                valuation.code.in_(etf_list)
            )
            df = get_fundamentals(q, date=target_date)
            
            if df is None or df.empty:
                logger.warning("  -> 警告: get_fundamentals(market_cap) 返回为空。")
                return pd.Series()
                
            # 建立映射: code -> market_cap
            # 注意: returns dataframe with columns [code, market_cap]
            mapping = df.set_index('code')['market_cap']
            return mapping
            
        except Exception as e:
            logger.error(f"  -> 获取市值出错: {e}")
            return pd.Series()
            
    else:
        logger.warning(f"  -> 警告: 未知的代表方法 '{method}', 默认为 'money'。")
        return get_history_data(etf_list, target_date, 30, "money").mean()


# -------------------- 1.7 内部助手函数：公共逻辑 --------------------------
def _prepare_data_and_corr(etf_list, config, target_date):
    """
    公共逻辑：预先过滤流动性 + 计算平滑相关性矩阵
    返回: (corr_matrix, final_etf_list)
    """
    if target_date is None: target_date = datetime.date.today()
    
    # 1. 预先过滤流动性
    money_median_20d = get_history_data(etf_list, target_date, 20, "money").median()
    valid_etfs = [etf for etf in etf_list if money_median_20d.get(etf, 0) > config["min_liquidity"]]
    
    if not valid_etfs:
        logger.debug("  -> 无 ETF 通过流动性过滤。")
        return None, []
        
    # 2. 计算平滑相关性矩阵
    corr_matrix, final_etf_list = get_smoothed_correlation(valid_etfs, config, target_date)
    
    return corr_matrix, final_etf_list

def _select_best_from_groups(config, groups, final_etf_list, target_date, group_labels=None, return_details=False):
    """
    公共逻辑：从分组中选择代表性 ETF
    参数:
      groups: list of list, e.g. [[etf1, etf2], [etf3]]
      group_labels: list of label names corresponding to groups, e.g. [0, 1] (可选)
      return_details: 是否返回 {etf: label} 映射
    返回: selected_etfs 或 (selected_etfs, details)
    """
    # 获取排序依据
    rep_method = config.get("represent_method", "money")
    ranking_data = get_ranking_data(final_etf_list, target_date, method=rep_method)
    
    selected_etfs = []
    
    logger.debug(f"  -> 正在从每个簇中选择最佳 {rep_method} ETF:")
    
    if ranking_data.empty:
        logger.warning(f"  -> 警告: 方法 '{rep_method}' 的排名数据为空/缺失。选择将依赖列表顺序（任意）。")
    
    for i, etfs in enumerate(groups):
        if not etfs: continue
        label = group_labels[i] if group_labels else i
        
        # 选出该组中 ranking_data 最大的
        # 如果 ranking_data 为空/不全, default=0, 对于同样都是0的情况, max取第一个
        best_etf = max(etfs, key=lambda x: ranking_data.get(x, 0))
        selected_etfs.append(best_etf)
        
        display_name = get_security_info(best_etf).display_name
        logger.debug(f"    簇 {label}: 从 {len(etfs)} 个候选者中选择了 {display_name} ({best_etf})。")
        
    if return_details:
        details = {}
        # 将结果映射回字典 {etf: label}
        # 如果是某些特殊情况(如DBSCAN噪声)，label可能需要特殊处理
        # 这里假设 groups 和 group_labels 是一一对应的
        for i, etfs in enumerate(groups):
            label = group_labels[i] if group_labels else i
            for etf in etfs:
                details[etf] = label
        return selected_etfs, details
        
    return selected_etfs

# -------------------- 2.1 AP 聚类筛选 (Affinity Propagation) --------------------------
def ap_clustering_filter(etf_list, config, target_date=None, return_details=False):
    """
    使用 Affinity Propagation (AP) 算法进行聚类筛选
    """
    logger.debug(f"第二步: Affinity Propagation 聚类 (Damping={config['ap_damping']})...")
    
    # 1. 公共准备
    corr_matrix, final_etf_list = _prepare_data_and_corr(etf_list, config, target_date)
    if corr_matrix is None or corr_matrix.empty: return []
    
    # 4. 执行 AP 聚类
    ap = AffinityPropagation(
        damping=config["ap_damping"],
        preference=config["ap_preference"],
        affinity='precomputed',
        random_state=42
    )
    ap.fit(corr_matrix)
    
    labels = ap.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    logger.debug(f"  -> AP 收敛为 {n_clusters_} 个簇。")
    
    # 5. 构建分组
    cluster_dict = defaultdict(list)
    for etf, label in zip(final_etf_list, labels):
        cluster_dict[label].append(etf)
        
    groups = list(cluster_dict.values())
    group_labels = list(cluster_dict.keys())
    
    # 6. 选择代表
    return _select_best_from_groups(
        config, groups, final_etf_list, target_date,
        group_labels=group_labels, return_details=return_details
    )

# -------------------- 2.2 层次聚类筛选 (Hierarchical Clustering) --------------------------
def hierarchical_clustering_filter(etf_list, config, target_date=None, return_details=False):
    """
    使用层次聚类筛选
    """
    logger.debug(f"第二步: 层次聚类 (相关性阈值={config['cluster_corr_threshold']})...")

    # 1. 公共准备
    corr_matrix, final_etf_list = _prepare_data_and_corr(etf_list, config, target_date)
    if corr_matrix is None or corr_matrix.empty: return []

    # 4. 转换为距离矩阵 & 聚类
    distance_matrix = np.sqrt(2 * (1 - corr_matrix))
    np.fill_diagonal(distance_matrix.values, 0)
    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method="average")

    # 5. 聚类划分
    dist_threshold = np.sqrt(2 * (1 - config["cluster_corr_threshold"]))
    clusters = fcluster(Z, dist_threshold, criterion="distance")
    
    n_clusters_ = len(set(clusters))
    logger.debug(f"  -> 收敛为 {n_clusters_} 个簇。")

    # 6. 构建分组
    cluster_dict = defaultdict(list)
    for etf, label in zip(final_etf_list, clusters):
        cluster_dict[label].append(etf)
        
    groups = list(cluster_dict.values())
    group_labels = list(cluster_dict.keys())

    # 7. 选择代表
    return _select_best_from_groups(
        config, groups, final_etf_list, target_date,
        group_labels=group_labels, return_details=return_details
    )

# -------------------- 2.3 MST 聚类筛选 (Minimum Spanning Tree) --------------------------
def mst_clustering_filter(etf_list, config, target_date=None, return_details=False):
    """
    使用最小生成树 (MST) 算法进行聚类筛选
    原理: 构建完全图 (权重=距离)，生成MST，切断最长的K-1条边，形成K个连通分量
    """
    logger.debug(f"第二步: MST 聚类 (目标簇数={config['mst_n_clusters']})...")

    # 1. 公共准备
    corr_matrix, final_etf_list = _prepare_data_and_corr(etf_list, config, target_date)
    if corr_matrix is None or corr_matrix.empty: return []

    # 3. 构建距离矩阵 (Distance Matrix)
    dist_matrix = np.sqrt(2 * (1 - corr_matrix))
    np.fill_diagonal(dist_matrix.values, 0) # 自身距离为0

    # 4. 构建完全图并生成 MST
    G = nx.Graph()
    G.add_nodes_from(final_etf_list)
    
    edges = []
    n = len(final_etf_list)
    for i in range(n):
        for j in range(i + 1, n):
            u = final_etf_list[i]
            v = final_etf_list[j]
            weight = dist_matrix.iloc[i, j]
            edges.append((u, v, weight))
    
    G.add_weighted_edges_from(edges)
    
    logger.debug("  -> 正在构建最小生成树...")
    mst = nx.minimum_spanning_tree(G)

    # 计算指标 (保持不变)
    total_weight = sum(d['weight'] for u, v, d in mst.edges(data=True)) 
    ntl = total_weight / (len(mst.nodes()) - 1)
    avg_path_len = nx.average_shortest_path_length(mst, weight='weight')
    
        logger.debug(f"  [市场结构] NTL: {ntl:.4f} (越低 = 越高风险/系统性)")
        logger.debug(f"  [市场结构] 平均路径: {avg_path_len:.4f}")
    
    # 5. 切断最长的 K-1 条边
    mst_edges = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    num_to_remove = config['mst_n_clusters'] - 1
    
    if num_to_remove > 0 and len(mst_edges) >= num_to_remove:
        edges_to_remove = mst_edges[:num_to_remove]
        mst.remove_edges_from([(u, v) for u, v, d in edges_to_remove])
        logger.debug(f"  -> 移除了 {num_to_remove} 条最长边以形成簇。")
    
    # 6. 获取连通分量 (即聚类结果)
    components = list(nx.connected_components(mst))
    logger.debug(f"  -> 生成了 {len(components)} 个簇。")
    
    # 7. 构建分组 (components本身就是sets的list)
    groups = [list(c) for c in components]
    group_labels = list(range(len(groups))) # MST没有标签，只有分量ID

    # 8. 选择代表
    return _select_best_from_groups(
        config, groups, final_etf_list, target_date,
        group_labels=group_labels, return_details=return_details
    )

# -------------------- 2.4 DBSCAN 聚类筛选 (Density-Based) --------------------------
def dbscan_clustering_filter(etf_list, config, target_date=None, return_details=False):
    """
    使用 DBSCAN 进行基于密度的聚类
    原理: 将高密度区域划分为簇，稀疏区域标记为噪声 (-1)
    """
    eps = config.get('dbscan_eps', 0.5) # 默认距离阈值
    min_samples = config.get('dbscan_min_samples', 2) # 最小样本数
    
    logger.debug(f"第二步: DBSCAN 聚类 (eps={eps}, min_samples={min_samples})...")

    # 1. 公共准备
    corr_matrix, final_etf_list = _prepare_data_and_corr(etf_list, config, target_date)
    if corr_matrix is None or corr_matrix.empty: return []

    # 3. 计算距离矩阵
    dist_matrix = np.sqrt(2 * (1 - corr_matrix))
    np.fill_diagonal(dist_matrix.values, 0)

    # 4. 执行 DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    db.fit(dist_matrix)
    
    labels = db.labels_
    
    # 统计簇
    unique_labels = set(labels)
    n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    logger.debug(f"  -> DBSCAN 发现了 {n_clusters_} 个簇以及 {n_noise_} 个噪声点。")

    # 5. 构建分组 (特殊处理: 噪声单独成组)
    cluster_dict = defaultdict(list)
    for etf, label in zip(final_etf_list, labels):
        cluster_dict[label].append(etf)
    
    groups = []
    group_labels = []

    # A. 处理普通簇
    for label in unique_labels:
        if label == -1: continue
        groups.append(cluster_dict[label])
        group_labels.append(label)
    
    # B. 处理噪声 (Outliers) - 每个噪声单独作为一个组(必选)
    # 或者直接把它们加到 selected_etfs 里，但为了复用逻辑，我们可以把它们视为单元素组
    if -1 in cluster_dict:
        noise_etfs = cluster_dict[-1]
        logger.debug(f"  -> 保留 {len(noise_etfs)} 个独立离群值 (噪声) 作为单独候选。")
        for etf in noise_etfs:
            groups.append([etf])
            group_labels.append(-1) # 标记为噪声组

    # 6. 选择代表
    return _select_best_from_groups(
        config, groups, final_etf_list, target_date,
        group_labels=group_labels, return_details=return_details
    )