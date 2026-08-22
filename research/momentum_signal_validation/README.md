# 动量信号验证（本地 P1）

本目录研究宽基、申万一级行业和风格指数的横截面动量。当前架构是：JQ 只导出原始输入，本地程序独立计算；活动流程不再运行 JQ 版 P1，也不再把历史 JQ 结果作为校正基准。

逐步研究路线、当前指数结果的证据解读和各阶段晋级条件见 [`docs/step_by_step_research_plan.md`](docs/step_by_step_research_plan.md)。当前 P1 决策是：风格指数进入 ETF 载体层，宽基只作对照，申万一级行业暂停。

## 模块边界

- [`adapters/jq/export_p1_index_inputs.py`](adapters/jq/export_p1_index_inputs.py)：唯一的 JQ 目标程序，负责获取日收盘价、catalog 和交易日历，并输出一个 `tar.gz`。它必须遵守项目级 [`JQ 运行时兼容性`](../../docs/reference/joinquant/jq_research_compatibility.md)。
- [`local/validate_p1_index_archive.py`](local/validate_p1_index_archive.py)：检查 tar/解压目录的成员路径、文件集合、字节数和 SHA-256。
- [`local/run_p1_from_snapshot.py`](local/run_p1_from_snapshot.py)：读取 manifest 和 CSV，重建完整指数面板，调用本地 P1，并写入版本化结果目录。
- [`p1_jq_signal_validation.py`](p1_jq_signal_validation.py)：文件名因既有引用暂时保留，内容已经是纯本地计算 module；不导入 `jqdata`、不取数、不写文件、不生成平台 runtime 校正信息。

旧 JQ 程序、旧 JQ 输出以及一次性的 JQ/本地对照均只存在于 [`archive/`](archive/README.md)，不被活动代码导入。

## 研究口径

三个截面独立排名：

| 截面 | 输入 catalog | 最小有效资产数 |
| --- | --- | ---: |
| `broad` | 10 个冻结宽基指数 | 6 |
| `industry_sw_l1` | 全部历史申万一级代码 | 20 |
| `style` | 12 个冻结风格指数 | 6 |

默认形成期为 10、15、20、25、30、40、60、90 个交易日，预测期为 1、3、5、10、20 个交易日。信号包括区间收益、对数价格年化斜率、R² 和 `slope_x_r2`；主单元仍是 `slope_x_r2 / L=25 / H=5`。

日期 `t` 的信号只使用截至 `close[t]` 的数据，未来收益定义为 `close[t+H] / close[t] - 1`。这是信号研究口径，不代表可以在 `close[t]` 成交。

## 数据导出

JQ Notebook 中执行完整导出器定义后运行：

```python
ARCHIVE_INFO = export_p1_index_inputs()
```

默认只生成一个文件：

```text
momentum_index_p1_inputs_v1_20150605_20260820.tar.gz
```

正式区间前仍应跑一次短区间冒烟，但冒烟只确认 JQ API、非空数据和打包路径可用，不承担结果校正。

## 本地运行

先检查已解压输入：

```bash
python -m research.momentum_signal_validation.local.validate_p1_index_archive \
  research/momentum_signal_validation/data/momentum_index_p1_inputs_v1
```

然后写入一个不存在的新目录：

```bash
python -m research.momentum_signal_validation.local.run_p1_from_snapshot \
  research/momentum_signal_validation/data/momentum_index_p1_inputs_v1 \
  --output research/momentum_signal_validation/data/local_results/momentum_index_p1_inputs_v1__p1_local_v2
```

本地 P1 只输出 11 张研究表：协议和覆盖、IC 汇总与主单元日明细、参数平台、分组及 Top-K、R² 双排序和年度主 IC。结果 manifest 记录输入 manifest SHA-256、本地 Python/pandas/numpy 版本和每张输出表的 checksum。

## 只保留的检查

活动流程只保留会阻止错误计算的检查：

1. 输入文件属于 manifest，且字节数和 SHA-256 一致。
2. catalog、交易日历和价格表含必要字段；日期/代码键不重复，价格日期不超出交易日历。
3. 输入包含最长形成期所需的前置历史，每个截面达到最低有效资产数。
4. 输出目录不得覆盖，结果表写入后记录 checksum。

以下逻辑已从活动流程删除：JQ/Python runtime hash 校正、JQ 与本地逐表比较、自动“通过/失败”协议门槛、为了兼容 JQ 而保留的 Python 3.6/旧 pandas 写法，以及 P1 内部的 JQ 取数和导出。

本地测试验证信号无前视、形成期/预测期语义、IC/HAC 计算、面板加载和上述少量完整性检查。测试通过不代表信号有效，研究判断直接基于本地结果表。
