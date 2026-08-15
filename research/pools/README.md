# ETF 资产池研究

本目录研究“从哪些 ETF 中轮动”，包括流动性过滤、同指数去重、相关性聚类、族谱追踪和 PCA 分析。它是研究工具集合，不是一个统一包。

## 主要代码

| 文件 | 用途 |
| --- | --- |
| `src/ap_pools.py` | JQ 环境中的 ETF 过滤、聚类和动量评分主实验 |
| `src/join_method.py` | AP、层次聚类、MST、DBSCAN 等聚类实现 |
| `src/pca_analysis.py` | PCA 与一致性分析 |
| `src/cluster_analysis.py` | 多期聚类族谱与桑基图生成，结果写入 `artifacts/cluster_results/` |
| `src/analyze_pkl.py` | 读取固定缓存的专项分析脚本，使用仓库相对路径 |
| `src/dynamic_pools.py`、`src/oix_pools.py` | 社区或早期资产池参考实现 |

## 研究文档与产物

- `src/`：JQ 资产池实验和本地专项分析脚本；直接运行时从此目录执行。
- `docs/`：资产池设计、阶段性结论和外部研究材料；`docs/assets/` 保存文档配图。
- `artifacts/cluster_results/`：缓存、图像、CSV、HTML 和历史压缩包；聚类脚本会在此处读写结果。

运行代码前先区分 JQ 数据接口和本地缓存路径。`src/analyze_pkl.py` 与 `src/cluster_analysis.py` 均使用项目内的 `artifacts/cluster_results/`，不依赖机器特定绝对路径。
