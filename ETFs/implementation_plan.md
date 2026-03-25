# 实施计划 - 性能优化 (深度集成 JQ API)

本计划旨在通过 JQ API 的批量处理能力和向量化计算，最大限度减少 API 调用次数并加速回测速度，同时严格保持原有策略逻辑。

## 核心优化策略

### 1. 证券基础信息批量获取与缓存
- **操作**: 在 `initialize` 中使用 `get_all_securities` 一次性获取所有 ETF 的名称及上市日期。
- **目的**: 消除运行中对 `get_security_info` 和 `get_current_data()[s].name` 的单标的查询。

### 2. 行情数据批量请求 (Batching)
- **操作**: 在 `get_final_ranked_etfs` 中，用一个 `get_price(pool, ...)` 替换原有的循环 `attribute_history`。
- **目的**: 合并网络请求（或数据库查询），解决回测中最主要的瓶颈。

### 3. 指标计算向量化 (Vectorization)
- **操作**: 使用 `numpy` 对批量获取的收盘价矩阵进行操作，一次性计算全池标的的动量、R2、RSI 等。
- **目的**: 利用 Python 库的底层加速，减少循环开销。

### 4. ATR 每日定时更新
- **操作**: 每日 09:10 计算全池 ATR 并存入 `g.atr_cache`。
- **目的**: 避免分钟级逻辑中重复进行日线级别的历史数据查询。

## 详细修改点

### [ETF_7star_opt.py](file:///c:/Users/84066/Documents/Quan/jqtrade/ETFs/ETF_7star_opt.py) [MODIFY]

- **[MODIFY] initialize**: 
    - 初始化 `g.security_info` (存储名称和上市日)。
    - 计算一次 `g.fixed_pool_info`。
- **[MODIFY] get_security_name**: 改为查表逻辑。
- **[MODIFY] get_final_ranked_etfs**:
    - 使用 `get_current_data()` 返回的字典进行全池批量状态检查。
    - 批量 `get_price` 获取所需长度的历史数据。
    - 将指标计算逻辑合并为对 DataFrame/ndarray 的操作。
- **[MODIFY] minute_level_atr_stop_loss**: 引用 `g.atr_cache`。

## 验证计划

- **逻辑等价测试**: 选取典型交易日，打印优化前后的 `momentum_score` 和筛选结果列表，确保误差在允许范围（浮点精度）内。
- **性能评估**: 回测一年历史数据，对比优化前后的秒级耗时。
