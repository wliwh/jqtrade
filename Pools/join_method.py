from jqdata import *
from jqfactor import *
import pandas as pd
import numpy as np
import datetime
import builtins
import math
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from collections import defaultdict

def get_history_data(security_list, end_date, count, field="close"):
    """
    Helper to fetch history data ending at a specific date using get_price.
    Handles strict formatting to return a DataFrame (Index=Date, Columns=Securities).
    """
    # Ensure input is a list
    if isinstance(security_list, str):
        security_list = [security_list]

    try:
        # User confirmed get_price with panel=False returns a flat DataFrame
        df = get_price(list(security_list), end_date=end_date, count=count, 
                       frequency='daily', fields=[field], panel=False)
        
        if df.empty:
            return pd.DataFrame()

        # 1. Handle Flat DataFrame (Multiple Securities) using Pivot
        if 'code' in df.columns:
            index_col = 'time' if 'time' in df.columns else 'date' if 'date' in df.columns else None
            
            if index_col:
                # Pivot to Index=Date, Columns=Code
                return df.pivot(index=index_col, columns='code', values=field)
            else:
                return df.pivot(columns='code', values=field)

        # 2. Handle Single Security (Index = Date)
        if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)) or (len(df) > 0 and isinstance(df.index[0], (datetime.date, datetime.datetime, pd.Timestamp))):
            if len(security_list) == 1:
                security_code = security_list[0]
                if field in df.columns:
                     return df[[field]].rename(columns={field: security_code})
            
            return df[field] # Fallback

        # 3. Legacy Panel or ndim=3
        if getattr(df, "ndim", 2) == 3:
             return df[field]
             
        # 4. MultiIndex (Date, Code) - New JQData sometimes
        if isinstance(df.index, pd.MultiIndex):
            unstacked = df.unstack(level=1)
            if field in unstacked.columns:
                return unstacked[field]
            if field in unstacked.columns.get_level_values(0):
                 return unstacked.xs(field, axis=1, level=0, drop_level=True)
            return unstacked
        return df[field]

    except Exception as e:
        print(f"  [Error] get_history_data failed: {e}")
        return pd.DataFrame()

# -------------------- 1.5 核心工具：平滑相关性计算 --------------------------
def get_smoothed_correlation(etf_list, config, target_date=None, verbose=True):
    """
    计算平滑后的相关性矩阵，以保证聚类结果的时序稳定性。
    逻辑：计算过去N个时间窗口的相关性矩阵平均值。
    """
    if target_date is None: target_date = datetime.date.today()
    
    # 1. 获取参数
    window = config["correlation_window"]
    steps = config.get("smoothing_steps", 1)  # 平滑步数，默认1(不平滑)
    lag = config.get("smoothing_lag", 20)     # 每次滞后的天数，默认20(约1个月)
    
    if verbose and steps > 1:
        print(f"  -> Computing smoothed correlation (Steps={steps}, Lag={lag}d)...")
        
    # 2. 计算所需总数据长度
    # 需要的数据长度 = 窗口 + (步数-1) * 滞后
    total_count = window + (steps - 1) * lag
    
    # 3. 获取数据
    price_data = get_history_data(etf_list, target_date, total_count, "close")
    price_data = price_data.fillna(method="ffill").dropna(axis=1, how="any")
    
    if price_data.empty:
        if verbose: print("  -> Price data empty.")
        return None, []
        
    # 计算所有日期的收益率
    all_returns = price_data.pct_change().dropna()
    valid_etfs = all_returns.columns.tolist()
    
    if verbose and len(valid_etfs) < len(etf_list):
        print(f"  -> Dropped {len(etf_list)-len(valid_etfs)} ETFs due to NaN data.")
        
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
            if verbose: print(f"  -> Warning: Not enough data for smoothing step {i+1}.")
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

# -------------------- 2.1 AP 聚类筛选 (Affinity Propagation) --------------------------
def ap_clustering_filter(etf_list, config, target_date=None, verbose=True, return_details=False):
    """
    使用 Affinity Propagation (AP) 算法进行聚类筛选
    """
    if verbose: print(f"Step 2: Affinity Propagation Clustering (Damping={config['ap_damping']})...")
    
    # 1. 预先过滤流动性 (减少计算量)
    if target_date is None: target_date = datetime.date.today()
    
    # money_median_20d = history(20, "1d", "money", etf_list, df=True).median()
    money_df = get_history_data(etf_list, target_date, 20, "money")
    money_median_20d = money_df.median()
    valid_etfs = [etf for etf in etf_list if money_median_20d.get(etf, 0) > config["min_liquidity"]]
    
    if not valid_etfs:
        if verbose: print("  -> No ETFs passed liquidity filter.")
        return []

    # 2. 计算平滑相关性矩阵 (替代原有的直接获取数据计算)
    corr_matrix, final_etf_list = get_smoothed_correlation(valid_etfs, config, target_date, verbose)
    
    if corr_matrix is None or corr_matrix.empty:
        return []
    
    # 4. 执行 AP 聚类
    # 注意：AP 接受相似度矩阵 (Similarity Matrix)，相关系数本身就是一种相似度 (-1 ~ 1)
    # affinity='precomputed' 表示我们直接传入矩阵
    ap = AffinityPropagation(
        damping=config["ap_damping"],
        preference=config["ap_preference"],
        affinity='precomputed',
        random_state=42
    )
    ap.fit(corr_matrix)
    
    labels = ap.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if verbose: print(f"  -> AP converged into {n_clusters_} clusters.")
    
    # 5. 从每个簇中选择流动性最好的 ETF
    
    # 准备成交额数据
    # money_data = history(30, "1d", "money", final_etf_list, df=True).mean()
    money_data = get_history_data(final_etf_list, target_date, 30, "money").mean()
    
    cluster_dict = defaultdict(list)
    for etf, label in zip(final_etf_list, labels):
        cluster_dict[label].append(etf)
        
    selected_etfs = []
    
    if verbose: print("  -> Selecting best liquidity ETF from each cluster:")
    for label, etfs in cluster_dict.items():
        # 选出该组中 money_data 最大的
        best_etf = max(etfs, key=lambda x: money_data.get(x, 0))
        selected_etfs.append(best_etf)
        
        # 可选：打印每组详情
        display_name = get_security_info(best_etf).display_name
        
        if verbose: print(f"    Cluster {label}: Selected {display_name} ({best_etf}) from {len(etfs)} candidates.")

    if return_details:
        # construct a dict {etf: label}
        # final_etf_list contains the ETFs used in clustering
        details = {etf: label for etf, label in zip(final_etf_list, labels)}
        return selected_etfs, details

    return selected_etfs

# -------------------- 2.2 层次聚类筛选 (Hierarchical Clustering) --------------------------
def hierarchical_clustering_filter(etf_list, config, target_date=None, verbose=True, return_details=False):
    """
    使用层次聚类筛选
    """
    if verbose: print(f"Step 2: Hierarchical Clustering (Corr Threshold={config['cluster_corr_threshold']})...")

    # 1. 预先过滤流动性
    if target_date is None: target_date = datetime.date.today()
    money_median_20d = get_history_data(etf_list, target_date, 20, "money").median()
    valid_etfs = [etf for etf in etf_list if money_median_20d.get(etf, 0) > config["min_liquidity"]]

    if not valid_etfs:
        if verbose: print("  -> No ETFs passed liquidity filter.")
        return []

    # 2. 计算平滑相关性矩阵
    corr_matrix, final_etf_list = get_smoothed_correlation(valid_etfs, config, target_date, verbose)

    if corr_matrix is None or corr_matrix.empty:
        return []

    # 4. 转换为距离矩阵 & 聚类
    # distance = sqrt(2 * (1 - correlation))
    distance_matrix = np.sqrt(2 * (1 - corr_matrix))
    # distance_matrix 对角线设为0 (虽然公式结果也是0，确保数值稳定)
    np.fill_diagonal(distance_matrix.values, 0)

    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method="average")

    # 5. 聚类划分
    # threshold = sqrt(2 * (1 - corr_threshold))
    dist_threshold = np.sqrt(2 * (1 - config["cluster_corr_threshold"]))
    clusters = fcluster(Z, dist_threshold, criterion="distance")
    
    n_clusters_ = len(set(clusters))
    if verbose: print(f"  -> Converged into {n_clusters_} clusters.")

    # 6. 从每个簇中选择流动性最好的 ETF
    money_means = get_history_data(final_etf_list, target_date, 30, "money").mean()

    cluster_dict = defaultdict(list)
    for etf, label in zip(final_etf_list, clusters):
        cluster_dict[label].append(etf)

    selected_etfs = []
    
    if verbose: print("  -> Selecting best liquidity ETF from each cluster:")
    for label, etfs in cluster_dict.items():
        # 选出该组中 money_means 最大的
        best_etf = max(etfs, key=lambda x: money_means.get(x, 0))
        selected_etfs.append(best_etf)
        
        display_name = get_security_info(best_etf).display_name
        if verbose: print(f"    Cluster {label}: Selected {display_name} ({best_etf}) from {len(etfs)} candidates.")

    if return_details:
        details = {etf: label for etf, label in zip(final_etf_list, clusters)}
        return selected_etfs, details

    return selected_etfs

# -------------------- 2.3 MST 聚类筛选 (Minimum Spanning Tree) --------------------------
def mst_clustering_filter(etf_list, config, target_date=None, verbose=True, return_details=False):
    """
    使用最小生成树 (MST) 算法进行聚类筛选
    原理: 构建完全图 (权重=距离)，生成MST，切断最长的K-1条边，形成K个连通分量
    """
    if verbose: print(f"Step 2: MST Clustering (Target Clusters={config['mst_n_clusters']})...")

    # 1. 预先过滤流动性
    if target_date is None: target_date = datetime.date.today()
    money_median_20d = get_history_data(etf_list, target_date, 20, "money").median()
    valid_etfs = [etf for etf in etf_list if money_median_20d.get(etf, 0) > config["min_liquidity"]]

    if not valid_etfs:
        if verbose: print("  -> No ETFs passed liquidity filter.")
        return []

    # 2. 计算平滑相关性矩阵
    corr_matrix, final_etf_list = get_smoothed_correlation(valid_etfs, config, target_date, verbose)
    
    if corr_matrix is None or corr_matrix.empty:
        return []

    # 3. 构建距离矩阵 (Distance Matrix)
    dist_matrix = np.sqrt(2 * (1 - corr_matrix))
    np.fill_diagonal(dist_matrix.values, 0) # 自身距离为0

    # 4. 构建完全图并生成 MST
    G = nx.Graph()
    # 添加节点
    G.add_nodes_from(final_etf_list)
    
    # 添加边 (完全图)
    edges = []
    n = len(final_etf_list)
    for i in range(n):
        for j in range(i + 1, n):
            u = final_etf_list[i]
            v = final_etf_list[j]
            weight = dist_matrix.iloc[i, j]
            edges.append((u, v, weight))
    
    G.add_weighted_edges_from(edges)
    
    if verbose: print("  -> Building Minimum Spanning Tree...")
    mst = nx.minimum_spanning_tree(G)

    # 1. 计算标准化树长 (NTL)
    total_weight = sum(d['weight'] for u, v, d in mst.edges(data=True)) 
    ntl = total_weight / (len(mst.nodes()) - 1)
    # 2. 计算平均路径长度 (需要算拓扑距离，或是加权距离)
    avg_path_len = nx.average_shortest_path_length(mst, weight='weight')
    # 3. 打印观测
    if verbose:
        print(f"  [Market Structure] NTL: {ntl:.4f} (Lower = Higher Risk/Systemic)")
        print(f"  [Market Structure] Avg Path: {avg_path_len:.4f}")
    
    # 5. 切断最长的 K-1 条边
    mst_edges = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    num_to_remove = config['mst_n_clusters'] - 1
    
    if num_to_remove > 0 and len(mst_edges) >= num_to_remove:
        edges_to_remove = mst_edges[:num_to_remove]
        mst.remove_edges_from([(u, v) for u, v, d in edges_to_remove])
        if verbose: print(f"  -> Removed {num_to_remove} longest edges to form clusters.")
    
    # 6. 获取连通分量 (即聚类结果)
    components = list(nx.connected_components(mst))
    if verbose: print(f"  -> Generated {len(components)} clusters.")
    
    # 7. 从每个簇中选择流动性最好的 ETF
    money_means = get_history_data(final_etf_list, target_date, 30, "money").mean()
    
    selected_etfs = []
    if verbose: print("  -> Selecting best liquidity ETF from each cluster:")
    
    for i, comp in enumerate(components):
        etfs_in_cluster = list(comp)
        # 选出该组中 money_means 最大的
        best_etf = max(etfs_in_cluster, key=lambda x: money_means.get(x, 0))
        selected_etfs.append(best_etf)
        
        display_name = get_security_info(best_etf).display_name
        if verbose: print(f"    Cluster {i+1}: Selected {display_name} ({best_etf}) from {len(etfs_in_cluster)} candidates.")
        
    if return_details:
        # Map back to labels. For MST, components are clusters.
        # Assign an arbitrary ID to each component
        details = {}
        for idx, comp in enumerate(components):
            for etf in comp:
                details[etf] = idx
        return selected_etfs, details

    return selected_etfs

# -------------------- 2.4 DBSCAN 聚类筛选 (Density-Based) --------------------------
def dbscan_clustering_filter(etf_list, config, target_date=None, verbose=True, return_details=False):
    """
    使用 DBSCAN 进行基于密度的聚类
    原理: 将高密度区域划分为簇，稀疏区域标记为噪声 (-1)
    """
    eps = config.get('dbscan_eps', 0.5) # 默认距离阈值
    min_samples = config.get('dbscan_min_samples', 2) # 最小样本数
    
    if verbose: print(f"Step 2: DBSCAN Clustering (eps={eps}, min_samples={min_samples})...")

    # 1. 预先过滤流动性
    if target_date is None: target_date = datetime.date.today()
    money_median_20d = get_history_data(etf_list, target_date, 20, "money").median()
    valid_etfs = [etf for etf in etf_list if money_median_20d.get(etf, 0) > config["min_liquidity"]]

    if not valid_etfs:
        if verbose: print("  -> No ETFs passed liquidity filter.")
        return []

    # 2. 计算平滑相关性矩阵
    corr_matrix, final_etf_list = get_smoothed_correlation(valid_etfs, config, target_date, verbose)
    
    if corr_matrix is None or corr_matrix.empty:
        return []

    # 3. 计算相关性矩阵 & 距离矩阵
    # DBSCAN metric='precomputed' 需要距离矩阵
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
    
    if verbose: print(f"  -> DBSCAN found {n_clusters_} clusters and {n_noise_} noise points.")

    # 5. 选择逻辑
    money_means = get_history_data(final_etf_list, target_date, 30, "money").mean()
    cluster_dict = defaultdict(list)
    
    for etf, label in zip(final_etf_list, labels):
        cluster_dict[label].append(etf)
        
    selected_etfs = []
    
    if verbose: print("  -> Selecting best liquidity ETF from each cluster:")
    
    # 处理普通簇
    for label in unique_labels:
        if label == -1:
            continue # 处理噪声单独逻辑
            
        etfs = cluster_dict[label]
        best_etf = max(etfs, key=lambda x: money_means.get(x, 0))
        selected_etfs.append(best_etf)
        
        display_name = get_security_info(best_etf).display_name
        if verbose: print(f"    Cluster {label}: Selected {display_name} ({best_etf}) from {len(etfs)} candidates.")
        
    # 处理噪声 (Outliers)
    # 策略: 噪声代表独特资产，全部保留
    if -1 in cluster_dict:
        noise_etfs = cluster_dict[-1]
        if verbose: print(f"  -> Retaining {len(noise_etfs)} unique outliers (Noise):")
        for etf in noise_etfs:
            selected_etfs.append(etf)
            # if verbose: print(f"     Outlier: {get_security_info(etf).display_name} ({etf})")
            
    if return_details:
        details = {etf: label for etf, label in zip(final_etf_list, labels)}
        return selected_etfs, details

    return selected_etfs