# JQ 研究环境兼容性

`export_all_a_breadth.py`、P1 V1 和 P1 V2 完整快照均已在 JQ 运行；V2 已通过本地验收，但最终导出代码仍缺一份独立短区间 JQ 重跑作交叉核验。

## 固定限制

- JQ 使用 Python 3.6、旧 pandas 和 `from jqdata import *`；本地 `jqdatasdk` 导入方式不能混用。
- 不使用 `Series.to_numpy(dtype=...)`、nullable boolean、`groupby(dropna=False)` 或 `to_csv(lineterminator=...)`；使用 `.values`、普通 `bool` 和基础 `groupby`。
- `000985.XSHG` 历史成分从 2012 年可用；默认请求 `2012-01-01`，首个交易日为 `2012-01-04`，不自动替换股票池。

## 平台冒烟

1. 记录 `sys.version`、`pd.__version__`，确认 `get_index_stocks('000985.XSHG', date='2012-01-04')` 非空。
2. 用约一个月区间完整生成并下载 ZIP；P1 需确认 `get_valuation` 的四个字段和每日非零换手分母。
3. 确认 manifest 为 `all_a_p1_inputs_v2`，含两个 MA 容差字段；不能重命名 V1 ZIP。
4. 修改函数后重跑完整定义单元或重启内核，再执行完整区间。

平台失败时保留 traceback；本地测试不能替代 JQ 数据覆盖、资源限制和运行结果。
