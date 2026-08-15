# JQ 研究环境兼容性备忘

本页记录 `jq_export_breadth.py` 在真实 JQ 投资研究环境中确认过的兼容性约束。编写或复制新的 JQ 研究脚本时先检查本页；本地较新的 Python/pandas 测试不能替代平台冒烟。

## 已确认的环境差异

- JQ 当前运行栈为 Python 3.6 和旧版 pandas；报错路径可见 `/opt/conda/lib/python3.6/`。
- JQ 研究脚本使用 `from jqdata import *`。不要在这类平台脚本中使用 `from __future__ import ...`。
- 上述导入规则只适用于 JQ 平台；本地 `jqdatasdk` 脚本仍使用 `from jqdatasdk import *`，两者不要混用。
- `000985.XSHG` 的历史成分在 JQ 中从 2012 年起可用。本研究默认从 `2012-01-01` 请求，首个实际交易日为 `2012-01-04`；2012 年以前不自动换用其他股票池。

## 旧 pandas 避坑

| 不要使用 | 兼容写法或处理 |
| --- | --- |
| `series.to_numpy(dtype=float)` | `np.asarray(series.values, dtype=float)` |
| `series.astype("boolean")` | 先 `fillna(False)`，再 `astype(bool)`；缺失状态另存掩码 |
| `groupby(..., dropna=False)` | 分组前显式过滤缺失键，再使用普通 `groupby(...)` |
| `to_csv(..., lineterminator="\n")` | 使用平台默认换行，不传该参数 |

新增 pandas 调用时，不能仅按本机版本判断是否存在；优先使用旧版长期可用的 `.values`、普通 `bool` 和基础 `groupby` 接口。

## 平台冒烟顺序

1. 在 JQ 中打印 `sys.version` 和 `pd.__version__`，保留实际版本信息。
2. 执行 `get_index_stocks('000985.XSHG', date='2012-01-04')`，确认结果非空。
3. 将导出日期缩短为一个月，完整执行脚本并检查 ZIP 可以下载、解压。
4. 修改 Notebook 中的函数后，重新执行完整定义单元或重启内核；只再次调用 `run_export()` 可能仍在使用内存中的旧函数。
5. 冒烟通过后再恢复完整日期范围。平台失败时保留完整 traceback，不用本地成功结果覆盖平台事实。

本地回归测试会检查 Python 3.6 语法，并禁止重新引入上表中的新版 API；真实 JQ 数据覆盖、资源限制和运行结果仍以平台实测为准。
