
# 2026年 项目进度简报

## 1. 策略开发 (ETFs/)
**状态**: 🟢 进行中 (优化与迭代)  
**概述**: 核心策略逻辑的存放地，包含多种基于 ETF 的轮动和择时策略。
- **已完成**:
    - `ETF_wy03.py`, `ETF_yj15.py`, `ETF_long.py`: 基础策略实现。
    - `ETF_modular.py`: 模块化框架搭建，支持策略组合。
- **进行中**:
    - `ETF_7star_modular.py`: 7star的模块化版本。
    - `momentum_scores.py`: 本地实现的动量评分。
- **待办**:
    [ ] 重点研究`gao`、`7star`两份策略的逻辑

## 2. 资产池构建 (Pools/)
**状态**: 🟡 活跃开发中 (聚类与筛选)  
**概述**: 负责 ETF 标的池的构建、筛选和动态维护，引入了聚类算法以降低资产相关性。
- **核心功能**:
    - `dynamic_pools.py`: 参考程序，取自joinquant社区。
    - `ap_pools.py`: 主程序，支持单次和多轮聚类筛选。
    - `join_method.py`：实现了 AP, Hierarchical, MST, DBSCAN 等多种聚类方法
    - `cluster_analysis.py`：实现了多轮聚类结果分析的工具，可本地运行
- **近期更新**:
    - **文档**: 更新了 `ap_pools_2.md`，详细记录了聚类分析的最新发现和工具使用指南。
    - **文档**: 更新了 `ap_pools_3.md`，优化过滤功能和桑。
- **下一步**:
    [x] 添加按指数过滤ETF（使得每个指数只保留一只代表ETF）的功能（`ap_pools.py`）
    [x] 需要将按交易额过滤ETF的代码移动到`ap_pools.py`中
    [x] 优化桑椹图，使其更加直观
    [x] 编写下一份文档，重点说明“按指数过滤ETF”这个功能
- **后期**：
    - 可集成`micro/`中的一致性分析（PCA等）

## 3. 回测引擎 (backtest_engine/)
**状态**: 🟢 稳定 (持续维护)  
**概述**: 提供策略回测、结果分析和性能评估的工具集。
- **核心组件**:
    - `Simple_Backtest_Manager.py`: 简易回测管理器。
    - `backtest_analyse.py`: 多次回测结果的对比分析与图表绘制。
    - `evaluate_pool.py`: 专门针对单次ETF轮动策略的评估，对比对象为ETF池内的标的，评估方法包括滚动收益对比、持仓段比较、换仓效果分析等。
- **文档**:
    - `JQ_backtest_API.md`: JoinQuant回测API说明文档，节略版。
    - `strategy_comparison*.md`: 策略对比分析报告。
- **观察**:
    - 目录结构清晰，功能划分明确。
    - `results/` 目录用于存放回测输出，需定期清理或归档。

## 4. 微观结构分析 (micro/)
**状态**: ⚪ 探索中 (早期阶段)  
**概述**: 关注市场微观结构、行业广度和一致性分析。
- **主要内容**:
    - `market_breadth.py`, `industries_breadth.html`: 市场/行业广度分析与可视化。
    - `eval_consistency.py`: 一致性评估工具。
- **文档**:
    - `micro_strategy_documentation.md`: 微观策略文档。
- **进度**:
    - 尚处于探索和工具开发阶段，代码量相对较少。
    - 侧重于数据分析和可视化展示。
