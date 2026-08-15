# 项目协作规则

## 入口与边界

- 先读根目录 `README.md`，再读目标子目录的 `README.md` 和相关文档。
- 文档检索先读 `docs/INDEX.md`，再按其中的领域边界使用 `rg -n -i -C 3` 搜索；不要整库加载文档。
- 现役策略只包括 `strategies/etf_rotation/README.md` 列出的 4 个文件。`strategies/etf_rotation/archive/` 与 `backtest_executor/archive/` 默认只读；除非任务明确涉及复现、迁移或修订历史实现，不要把归档代码重新接回现役入口。
- 当前开发计划以 `docs/planning/strategy_research_automation_pipeline_design.md` 为准。`docs/archive/` 中的内容不是当前工程事实源。

## 运行环境

- ETF 策略和真实参数回测运行在 JQ 研究环境，依赖平台注入的 `jqdata`、`create_backtest()` 和 `get_backtest()`。
- 编写、审阅或排查 JQ 策略时，先查询 `docs/reference/joinquant/README.md` 和其中的 `official/strategy-api.md`；数据、因子和优化器问题再查相应官方模块。本地快照与官网或实际 JQ 环境冲突时，以官网和实测为准。
- 本地测试只验证参数生成、代码注入、结果分析和数据导出等可离线部分，不能据此声称 JQ 策略回测通过。
- 性能或策略比较必须记录策略路径、逻辑 hash、参数、回测区间、成本口径和 JQ 回测 ID。

## 修改约束

- 新策略先通过源码审计和短区间冒烟，再进入参数搜索；不要直接批量提交未经验证的社区代码。
- `backtest_executor/config/` 只放现役策略配置；旧策略配置移入 `backtest_executor/archive/config/`。
- 配置中的策略路径必须存在。移动策略时同步检查 YAML、示例代码、结果元数据和论坛文章引用。
- `backtest_executor/results/` 是实验事实；不要为了整理目录改写既有 `mapper.json`、CSV 或回测 ID。
- `research/pools/artifacts/`、`research/micro/artifacts/` 和 `docs/reports/forum/assets/` 中的 CSV、HTML、图片是研究产物或文章资产，不是通用程序入口。

## 验证

- 普通 Python 改动至少运行相关测试；跨模块改动运行 `pytest -q`。
- 文档或目录整理至少检查 Markdown 链接、配置路径、Python 语法和 `git diff --check`。
- 无法在 JQ 环境验证时，明确区分“本地验证通过”和“JQ 实盘/回测未验证”。
