# 策略参数优化工具改进方案

为了进一步提升工具在 JQ 平台环境下的易用性与稳定性，在充分考虑平台限制（如无 `extras` 支持、并发限制等）的基础上，提出以下改进方案。

## 1. 架构与工作流整合 (Workflow)

*   **建立统一入口 (Orchestrator)**
    *   **职责定义**：创建一个 `optimize.py` 作为全局主控中心。
    *   **具体步骤**：
        1.  **加载配置**：解析用户提供的 `strategy_config.yaml`。
        2.  **生成空间**：调用寻优算法引擎（网格、阶段寻优等）生成参数全集。
        3.  **串行调度**：循环调用 `BacktestExecutorV3` 提交并监测回测。
        4.  **自动报表**：回测全部结束后，自动调用 `analyze_simple` 生成对比表格并存盘。
*   **基于“Experiment”的持久化管理**
    *   **目录结构规范**：
        ```text
        results/
        └── {strategy_name}_{timestamp}/
            ├── config_snapshot.yaml  # 本次运行的配置备份
            ├── mapper.json           # 存储 MD5 -> BacktestID 的持久化表
            ├── optimization_report.csv # 最终性能报表
            └── logs/                 # 详细运行日志
        ```
    *   **优势**：支持多次实验并行不干扰，且任意一次实验都可根据目录下的 `config_snapshot.yaml` 完全复现。
*   **添加更灵活的参数寻优方法 (Flexible Optimization Methods)**
    *   **方案**：支持多种寻优算法组件化，包括：
        1.  **Grid Search (网格搜索)**：全量覆盖搜索空间。
        2.  **Random Search (随机搜索)**：对大空间进行高效采样，快速定位可行域。
        3.  **Coordinate Descent (多轮迭代/坐标下降)**：支持多阶段优化。例如：第一轮寻优 A, B 参数并固定；第二轮在 A, B 最优的基础上寻优 C, D。
    *   **结果驱动与交互式寻优 (Interactive & Results-Driven)**：
        1.  **阶段暂停 (Stage Pause)**：支持在每轮寻优任务结束后自动暂停，生成中间分析报表。用户审查表现最好的若干组参数后，通过修改配置决定下一阶段的搜索路径。
        2.  **自动最优传递 (Auto-Passdown)**：支持配置自动从上一轮选取 `sort_by` 排名第一的参数值作为下一轮的 `fixed_params`。
    *   **实现**：通过 `SearchEngine` 接口进行抽象。用户在配置文件中指定 `engine: coordinate`，并利用 `steps` 数组定义每个阶段的策略。

## 2. 参数生命周期管理 (Parameter Management)

*   **外部配置驱动与命名映射 (Naming Mapping)**
    *   **方案**：在 YAML 配置文件中定义**三位一体**的命名映射：
        1.  `display_name`: 用于报表展示 (如 "评分范围")。
        2.  `short_id`: 用于生成 ID 和文件名 (如 "S")，保持简洁。
        3.  `strategy_var`: 策略代码中的全名 (如 `EXECUTION_SCORE_RANGE`)。
    *   **效果**：用户只需在配置中定义一次关系，系统在生成代码时自动映射到全名，在生成报表时映射到展示名，彻底解决“要记两套名字”的痛苦。
*   **唯一 ID 体系与断点续跑**
    *   对参数字典进行 MD5 哈希作为该组回测的唯一标识符。
    *   程序自动比对 `mapper.json`，跳过已完成的任务 ID，支持意外中断后的“零成本”续跑。
    *   确定json记录表的设计格式：
        ```json
        {
          "metadata": { "strategy": "ETF_Strategy", "experiment_time": "2026-03-10" },
          "runs": {
            "md5_config_hash": {
              "id_name": "S60_v06",
              "bt_id": "6e21bb...",
              "status": "done",
              "config": { "S": [0.0, 6.0], "v": [true, 5, 0.6] },
              "metrics": { "Calmar": 2.1, "MaxDD": -0.15 } 
            }
          }
        }
        ```

## 3. 回测执行与代码注入 (Execution)

*   **头部代码注入方案 (Header Injection)**
    *   **背景**：JQ 平台的 `extras` 参数不可用，且原有的“逐行正则替换”源码逻辑脆弱。
    *   **方案**：在生成回测代码时，将参数定义块（如 `EXECUTION_SCORE_RANGE = ...`）统一注入到代码最顶端。
    *   **优势**：不受原策略文件格式、注释或缩进的干扰，注入逻辑绝对可靠。
*   **串行执行模式 (Serial Execution)**
    *   **现状适配**：由于 JQ 平台对账号并发数的严格限制，继续沿用串行执行模式。
    *   **优化点**：每完成一组回测，并确认它的状态为done，再将 ID 存盘。

## 4. 统计与分析扩展 (Analysis Layer)（*暂时不修改这里*）

*   **网络容错与本地缓存**
    *   **问题**：API 请求失败会导致报表缺行，重复分析需重复请求。
    *   **方案**：增加带重试机制的 API 请求装饰器。获取到的性能指标立即本地缓存。如果 `mapper.json` 中的 ID 已有缓存数据，则直接读取，不再请求平台。
*   **多维度筛选与异常标注**
    *   **问题**：全量输出结果在参数空间大时难以阅读。
    *   **方案**：增加结果过滤参数（如 `--min-calmar 1.5`）。引入“参数平原”检查，自动标注那些指标虚高但在参数微调后表现剧降的“孤点（过拟合）”结果。
