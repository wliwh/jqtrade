# 项目全景、数据准备状态与重构上下文（外部工程快照）

> 归档于 2026-08-15。本文对应 `tdx-next-data`，不是 `jqtrade` 的项目上下文；文中的 Git HEAD、包结构和数据状态不可用于本仓库。

更新日期：2026-08-10<br>
适用基线：当前工作区（包含尚未提交的审计并行化、修订候选和相关文档改动）<br>
项目版本：`tdx-next-data 0.2.0`<br>
当前 Git HEAD：`42063e1 add：20260731数据冻结，准备审查`

## 1. 本文用途

本文是本项目的单文件上下文入口，供用户、开发者和在线模型快速理解：

- 项目要解决什么问题，第一阶段明确不做什么；
- 原始数据、审计、修订、canonical、快照和回测分别准备到什么程度；
- 仓库中哪些能力已经实现并经过测试，哪些只是组件或设计；
- 当前阻断正式数据发布的问题；
- 后续实现顺序、不可破坏的不变量和适合重构的边界。

本文汇总的是 2026-08-10 的工作区事实。网页端模型可以只读取本文完成项目理解、总体设计和重构规划，不需要同时取得其他项目文档。若模型同时拥有仓库或数据目录访问权，发生冲突时按以下优先级判断：

1. 当前代码与配置；
2. 数据目录中的不可变 manifest、哈希和审计产物；
3. 本文。

## 2. 一句话结论

项目已经完成“原始历史数据准备、TDX 冻结、四项全量审计、问题日级候选生成”这半条链路，但尚未完成“问题决策清零、正式注册表重建、canonical 1 分钟构建、5 分钟/日线派生、发布质量门和首个不可变快照”这半条链路。

因此当前状态是：**原始数据和审计证据较完整，正式可回测数据尚未产出。**

磁盘上没有 `latest` 正式快照，也没有完整 canonical 对象集。现有唯一 `bars_1m` 对象只有 240 行、13,927 字节，是单标的 JQ 导入冒烟产物，不能用于正式回测。

## 3. 第一阶段目标与范围

### 3.1 目标

第一阶段建设可复现的全市场 ETF/LOF 历史数据和回测基础：

- canonical 基础频率为 1 分钟；
- 覆盖当前上市及历史退市、终止、转型或合并标的；
- 正式 5 分钟和日线只从最终 1 分钟派生；
- 回测只读取固定截止日、不可变、可追溯的正式快照；
- 数据源、修订、排除、策略、参数和执行口径均可追溯。

当前候选快照截止日固定为 `2026-07-31`。

### 3.2 明确不在第一阶段

- 盘中实时行情；
- QMT、自动下单和真实交易执行；
- `tdxrs` 在线补洞或更新；
- 腾讯、东方财富等网页行情进入正式数据链；
- 回测期间联网或直接读取 raw JQ/TDX 文件；
- 为了抹平微小差异而静默改写来源值。

### 3.3 数据权威关系

| 数据 | 来源 | 当前规则 |
| --- | --- | --- |
| 历史 1 分钟 | JQ 主数据集及货币 ETF 补充集 | 回填 TDX 不覆盖的历史，并提供重叠核验 |
| TDX 覆盖区及以后 1 分钟 | 本地 TDX `.lc1` | 单日通过质量门后优先，不能无审计覆盖 |
| 正式 5 分钟 | canonical 1 分钟派生 | `.lc5` 只作参考 |
| 正式日线 | canonical 1 分钟派生 | `.day` 只作日聚合、状态和冲突核验 |
| 公司行为/复权 | 独立事件层 | 不改写 raw 未复权行情；当前尚未实现 |

来源选择原则上以 `symbol + trade_date` 为单位，避免在同一交易日无意混合两套分钟路径。

## 4. 当前端到端完成度

| 阶段 | 状态 | 已有结果 | 尚缺内容 |
| --- | --- | --- | --- |
| 需求与数据口径 | 已确认 | 第一阶段范围、截止日、来源优先级、分钟/日线语义已记录 | 公司行为和回测执行细则仍需落成代码/数据模型 |
| JQ 原始数据准备 | 已完成 | 两套数据集、ZIP、解压 CSV、state、inventory、哈希均保留 | 不能原样晋升，需应用审计决定 |
| TDX 原始输入冻结 | 已完成 | `tdx-capture-20260731-v1`，6,203 文件，约 2.2 GiB | 月度后继 capture 流程尚未端到端自动化 |
| 标的范围审计 | 已完成 | 2,015 只截止日有效正式重叠标的 | 正式注册表尚未按该结果重建 |
| JQ 独立质量审计 | 已完成但未通过质量门 | 657,340,080 行全部核验；发现严重内容异常 | 大量异常仍需批量规则或例外决定 |
| TDX 独立质量审计 | 已完成、仍需规则确认 | 无结构性致命错误；发现聚合和取整类复核项 | 容忍/接受规则尚未固化到 canonical 构建 |
| JQ/TDX 联合审计 | 已完成 | 129,013 个共同日已分类，无未解释缺分钟窗口 | 49 个真实日聚合差异和 44 个特殊开盘差异待归因 |
| 修订候选准备 | 已完成 | 8,853 个日级候选、冻结配置和 manifest | 8,830 个仍是阻断项 |
| 修订决定 | 刚开始 | 21 个 `accepted_minor`，2 个 `scope_excluded` | 其余候选未决定；尚无 v3 |
| canonical 构建 | 未实现 | 有 schema、读取器、合并、Parquet 存储等底层组件 | 缺来源选择、决定应用、日级隔离和应用记录编排 |
| 5 分钟派生 | 组件已实现 | `derive_5m()` 有单元测试 | 未接入真实 canonical 构建和发布链 |
| 日线派生 | 未实现 | 目标聚合口径已设计 | 没有 `derive_1d()` 或对应构建流程 |
| 发布质量门 | 未实现 | 已有基础 `validate_bars()` 和对象哈希检查 | 缺生命周期、排除、父快照差异、跨频率等完整门禁 |
| 正式快照 | 未发布 | 快照仓库类可写/查/验对象存在性 | 磁盘无 `snapshots/`，无 `manifests/latest.yaml` |
| 回测数据访问 | 组件已实现 | 快照裁剪读取、回测视图、NumPy mmap 缓存有单测 | 尚无真实快照可读，严格排除语义未实现 |
| 公司行为/复权/账户事件 | 未实现 | 只有需求和架构约束 | 事件数据、复权因子、读取视图均缺失 |
| VectorBT/RQAlpha 验收 | 未实现 | 环境文件已准备 | 框架适配器、共享策略核心和跨框架验收缺失 |

## 5. 当前数据资产与确定统计

### 5.1 数据目录

大体量数据不在 Git 仓库中，配置根目录为：

```text
~/.local/share/tdx-next-data
```

2026-08-10 实际占用约 54 GiB：

| 目录 | 约占用 | 内容 |
| --- | ---: | --- |
| `raw/joinquant` | 48 GiB | JQ ZIP、状态、inventory、解压 CSV |
| `raw/tdx_local` | 2.2 GiB | 冻结 TDX capture |
| `validation` | 305 MiB | 四项审计、历史校准轮次、修订候选 |
| `objects` | 24 KiB | 只有一份 240 行冒烟导入对象 |
| `registry` | 364 KiB | 旧注册表及历史版本 |

### 5.2 JQ 数据

| 数据集 | 文件与行数 | 状态 |
| --- | --- | --- |
| `jq-etf-lof-1m-20260731` | 53 ZIP、2,268 CSV、642,297,600 行 | 结构完整，内容质量门失败 |
| `jq-mmf-1m-20260731` | 2 ZIP、27 CSV、15,042,480 行 | 结构完整，内容质量门失败 |
| 合计 | 2,295 CSV、657,340,080 行 | 已全量审计，不能直接导入 canonical |

JQ 主集主要严重项：

- `invalid_ohlc`：866 个标的、11,235 条；
- `nonpositive_price`：183 个标的、3,667 条；
- `numeric_sentinel`：8 个标的、3,491 条；
- `negative_volume`：1 个标的、240 条；
- `negative_money`：1 个标的、240 条。

货币 ETF 补充集还有 21 只、959 条 `invalid_ohlc`，并存在隐含成交价与量额符号复核项。

### 5.3 TDX 冻结输入

首个正式输入批次：

```text
audit_batch_id = audit-batch-20260731-v2
capture_id     = tdx-capture-20260731-v1
as_of_date     = 2026-07-31
files          = 6,203
bytes          = 2,272,594,049
```

TDX 审计覆盖 2,105 个具有 catalog 或本地文件的标的。无重复/倒序时间戳、时段外记录、非正价格、非法 OHLC、负成交量或负成交额等结构性致命错误，但有：

- 1,265 只、5,978 天 `.lc1` 聚合与 `.day` 不一致；
- 1,021 只、43,163 条分钟成交量/额符号不一致；
- 部分极小成交表现为成交量取整到零但成交额非零；
- 缺少 `.day` 或 `.lc5` 的标的均不属于截止日有效集合，不阻断当前正式范围。

这些结果说明 TDX 是默认权威源，但仍必须按日经过质量门，不能无条件覆盖 JQ。

### 5.4 标的范围与特殊标的

范围审计识别 2,379 个历史或当前候选：

| 集合 | 数量 |
| --- | ---: |
| 截止日 JQ/TDX 重叠 | 2,015 |
| 截止日前已退市 | 278 |
| TDX catalog 有记录但无截止日分钟覆盖 | 72 |
| 截止日后上市 | 12 |
| override 明确排除 | 2 |

特殊决定：

- `501006.XSHG` 是未上市 C 类份额，全部日期从场内 universe 排除；这属于范围决定，不是坏数据排除。
- `501023.XSHG` 只保留到 2021-10-11 的场内历史；其后停牌骨架、终止和清算期记录不进入 canonical。终止日前另有 65 个实质参考差异日仍需处理。

### 5.5 两源联合审计

联合审计覆盖 2,015 只正式重叠标的、129,013 个交易日和 30,670,080 条对齐分钟记录：

| 分类 | 交易日数 | 含义 |
| --- | ---: | --- |
| `equivalent` | 73,169 | 允许误差内等价 |
| `intraday_rebucket` | 48,586 | 分钟分桶不同，但日聚合一致 |
| `open_field_mismatch` | 5,988 | 仅开盘字段语义不同 |
| `aggregate_mismatch` | 49 | 日级价格或量额仍不一致，需逐日归因 |
| `no_trade_placeholder_gap` | 1,221 | JQ 全零占位、TDX 无分钟记录的状态表示差异 |
| `incomplete_window` | 0 | 没有剩余的无法解释缺分钟窗口 |

5,988 个开盘差异中，TDX 日线支持 TDX 分钟开盘的有 5,944 天，支持 JQ 的有 44 天。后 44 天已进入修订候选，不能按默认 TDX 规则静默处理。

### 5.6 修订候选

当前最新运行是 `revision-candidates-20260731-v2`：

| 类别 | 数量 |
| --- | ---: |
| JQ 固有严重异常日 | 8,693 |
| 两源重叠差异 | 93 |
| `501023.XSHG` 特殊参考差异 | 65 |
| 范围/生命周期决定 | 2 |
| 合计 | 8,853 |

动作分布：

- `accepted_minor`：21；
- `scope_excluded`：2；
- `quarantine_pending`：8,830。

`quarantine_pending` 会阻止正式发布。候选按 `symbol + trade_date` 汇总，一项可同时命中多个规则，因此规则计数不可直接相加为候选数。

## 6. 当前代码结构与已实现能力

### 6.1 `src/tdx_data`：基础数据组件

| 文件 | 已实现能力 | 当前边界 |
| --- | --- | --- |
| `config.py` | 统一加载数据源、截止日、存储和容忍阈值 | 配置没有 canonical 构建/发布专用模型 |
| `schema.py` | canonical bar Arrow schema，保留来源、raw 值和修订字段 | schema 还没有独立质量标记/排除引用字段 |
| `sources/joinquant.py` | JQ inventory、CSV 分块读取、字段规范化和来源追踪 | reader 本身不消费审计决定 |
| `sources/tdx_local.py` | 原生解析 `.day/.lc1/.lc5`，不依赖 `tdxrs` | `read_minute()` 当前不会自动写入 capture ID |
| `freeze.py` / `audit_batch.py` | TDX 文件冻结、哈希、批次配置和复用 capture | 用于审计准备，不是月度发布编排器 |
| `registry.py` | 稳定 instrument ID、生命周期字段、override 和覆盖状态 | 主构建路径只读取一个 JQ `state.json`，见第 8 节 |
| `storage.py` | 分区写入、内容寻址 Parquet、只读对象 | 只提供底层写入器，不决定何为 canonical |
| `validation.py` | 键唯一、基本 OHLCV 合法性、简单跨频率摘要比较 | 不是完整发布质量门 |
| `merge.py` | 新旧帧冲突检测、容忍比较、重分桶分类 | 旧式全局 `accept_revisions` 模型不应作为新发布逻辑 |
| `derive.py` | A 股交易时段 1m→5m 聚合 | 没有 1m→1d；未处理批准排除清单 |
| `snapshots.py` | manifest 发布、latest、对象存在/哈希校验 | 会强制写 `status: complete`，缺发布门和排除状态 |
| `reader.py` | 按快照、频率、标的、时间和字段裁剪 Parquet | 不识别 approved exclusion，尚无严格缺口模式 |
| `backtest.py` | 回测投影视图、分区对象和 NumPy mmap 缓存 | 尚未对真实正式快照运行 |
| `pipeline.py` | 旧式注册表重建和 JQ 单数据集导入 | 不是当前目标架构的端到端 pipeline |

### 6.2 `src/tdx_data_audit`：独立审计包

已实现并在真实冻结批次上运行：

- `universe.py`：主 JQ、补充 JQ、TDX catalog/文件和 override 的范围联合审计；
- `joinquant.py`：JQ 文件、时间结构、日历、OHLCV 和异常规则的全量审计；
- `tdx_local.py`：`.day/.lc1/.lc5` 独立审计、跨频率比较和截止日口径；
- `overlap.py`：JQ/TDX 实际重叠区逐日分类；
- `execution.py`：最多 4 worker 的受控多进程执行；
- 中间结果带逻辑版本和来源指纹，可安全复用并确定性汇总。

审计通过 `scripts/audit_*.py` 运行，刻意不放入主 `tdx-data` CLI，也不写 canonical、注册表或快照。

### 6.3 `src/tdx_data_revision`：修订候选准备

当前只实现 `candidates.py`，能力包括：

- 校验审计 manifest 是否绑定同一批次、capture 和截止日；
- 从 JQ 严重异常、联合差异、特殊参考复核和生命周期规则生成日级候选；
- 为候选生成稳定 ID；
- 应用版本化 YAML 决定；
- 输出不可变 `candidates.parquet`、`unresolved.csv`、summary、冻结配置和 manifest；
- 拒绝原地覆盖同一运行 ID。

它**不读取并物化完整行情，不执行来源替换，不生成 canonical，不发布快照**。

### 6.4 当前命令面

主 CLI `tdx-data` 只有：

```text
status
jq-import
registry-build
snapshot-list
snapshot-show
snapshot-verify
```

独立脚本有：

```text
prepare_jq_archives.py
prepare_audit_batch.py
audit_universe.py
audit_jq_data.py
audit_tdx_data.py
audit_jq_tdx_overlap.py
generate_revision_candidates.py
```

目前不存在以下正式命令：

```text
tdx-import / canonical-build / derive-1d / candidate-validate
snapshot-publish / update-from-capture / corporate-actions-build
```

## 7. 测试和验证基线

2026-08-10 在 `tdx-next` Conda 环境运行：

```bash
PYTHONPATH=src /home/ai0/anaconda3/envs/tdx-next/bin/python -m pytest -q
```

结果：`34 passed in 0.58s`。

覆盖重点包括：

- JQ 读取和分区对象写入；
- TDX `.day/.lc1` 解码和日线成交量溢出规则；
- 注册表范围、override、历史退市和未来上市；
- TDX capture/审计批次准备；
- JQ、TDX、范围和联合审计核心分类；
- 多进程任务辅助逻辑；
- 修订候选生成、决定应用和不可变输出；
- 1m→5m 聚合；
- 冲突隔离/旧式修订开关；
- 快照 manifest、裁剪读取、回测视图和 mmap 缓存。

需要正确理解该测试结果：它证明已有组件的单元/小型集成行为，不证明 657M 行数据已经完成 canonical 构建，也不证明首个快照可发布。全量审计结果来自磁盘 manifest；本次文档整理没有重跑耗时全量审计。

## 8. 已知实现缺口与技术债

### 8.1 生产 pipeline 没有真正支持多 JQ 数据集

配置和审计层支持 `supplemental_dataset_roots`，但：

- `pipeline.import_joinquant()` 只构造 `JoinQuantDataset(config.joinquant.dataset_root)`；
- `pipeline.rebuild_registry()` 只把主数据集的 `state.json` 传给 `build_registry()`。

因此主 CLI 的导入/注册表路径没有完整消费货币 ETF 补充集。重构时应让“数据集集合”成为生产构建的一等输入，不能只修审计层。

### 8.2 正式注册表是旧基线

磁盘上的 `registry-20260808T084851523218Z` 显示：

```text
1,929 overlap + 62 JQ-only + 24 TDX-only + 2 pending
```

它生成于补数和最终范围审计之前。当前正确的截止日正式重叠集合是 2,015 只，但尚未发布为新注册表。任何 canonical 构建都必须先解决多数据集注册表问题并重建。

### 8.3 旧式 merge 开关不符合目标修订模型

`validation.accept_revisions` 是全局布尔值；一旦开启，会接受所有超过容忍阈值的来源冲突。这与“按规则、按日、按例外、全程有证据”的目标不一致，必须保持 `false`。

新的 canonical builder 应消费显式决定，不应复用该开关作为批量授权。

### 8.4 快照 API 早于当前发布设计

`SnapshotRepository.publish()` 当前：

- 不执行候选质量门；
- 不校验 blocking candidates；
- 不保存/校验批准排除；
- 不校验注册表、审计、修订运行与对象的批次一致性；
- 不支持 `validated_with_exclusions`，而是强制写 `complete`；
- 不验证父快照历史差异。

它只能视为底层 manifest 原子发布原型，不能直接暴露给首个正式构建流程。

### 8.5 日线与状态模型未落地

正式日线要求只聚合实际成交分钟：`volume > 0 or amount > 0`。全天无成交时保存状态，但不生成日线行情。当前没有 `derive_1d()`，也没有独立的日状态数据集。

JQ 全零分钟占位与 TDX 无记录的统一表达需要在 canonical builder 中明确，否则容易把状态误当行情或把无成交误判为缺失。

### 8.6 approved exclusion 尚无贯通实现

设计允许明确、经批准、无法可靠修复的 `symbol + trade_date` 从正式数据中排除，并令快照状态变为 `validated_with_exclusions`。目前候选动作集合支持该概念，但：

- 没有物化 exclusions 的 builder；
- 派生层不会阻止被排除日生成 5m/1d；
- snapshot manifest 没有排除清单契约；
- reader 没有默认严格报错机制；
- 回测运行记录也不能保存排除清单 ID。

### 8.7 数据 provenance 仍需统一

`BAR_SCHEMA` 已包含 `capture_id`、`ingest_id`、`source_ref` 等字段，但 TDX reader 当前把 `capture_id` 写为 `None`。canonical 构建必须保证每条来源记录能追溯到冻结 capture/dataset 和实际决定。

### 8.8 独立修订包已经成为当前实现边界

当前工作区已经存在独立的 `tdx_data_revision` 包，并已在真实冻结批次生成候选产物。后续重构应把它视为当前实现边界：继续保持候选/决定逻辑与 raw 解码、canonical 物化和快照发布解耦。不要依据更早的“首版不建设独立修订包”设想把它重新并回主 pipeline。

### 8.9 测试尚缺的关键场景

- 多 JQ 数据集贯穿注册表和生产导入；
- 从冻结 raw + 决定构建真实小型 canonical fixture；
- 日线派生、全天无成交和生命周期截断；
- approved exclusion 从候选到 reader 的端到端行为；
- 发布失败不切换 `latest`；
- 父快照历史差异门；
- 大批量分区、内存、断点恢复和确定性重建；
- 真实首版快照的集成/验收测试。

## 9. 后续工作建议顺序

### P0：先清理数据决定，解除发布阻断

1. 对数值哨兵、非正价格、负量额生成明确 `select_source` 或 `approved_exclusion` 决定；
2. 校准 `invalid_ohlc` 的差值、时点、成交状态分布，形成可批量验证的 `accepted_minor` 规则；
3. 归因 49 个 `aggregate_mismatch`、44 个特殊开盘差异；
4. 归因 `501023.XSHG` 的 65 个实质参考差异日；
5. 固化 TDX 极小成交取整和无成交占位规则；
6. 生成 revision candidates v3，目标是 `quarantine_pending = 0`，或只剩明确不能发布的系统性问题。

### P1：实现最小 canonical builder

建议新增一个明确的构建服务，而不是继续扩展旧 `merge_frames()`：

```text
冻结 raw + 新注册表 + 审计 manifests + revision decisions
  -> 验证输入批次一致性
  -> 按生命周期裁剪范围
  -> 按 symbol + trade_date 选择来源
  -> 应用 accepted_minor / select_source / approved_exclusion
  -> 输出 canonical 1m + applied/exclusions/unresolved
```

首版应优先做到确定、可恢复、可审计；复杂通用规则引擎和过早分区优化不是前置条件。

### P2：派生与发布质量门

1. 实现 `derive_1d()` 和无成交状态表示；
2. 将现有 `derive_5m()` 接入排除和生命周期语义；
3. 实现候选质量报告：键、时区、交易时段、生命周期、未决项、排除、跨频率、TDX 参考和对象哈希；
4. 实现父快照历史差异门；
5. 扩展 snapshot manifest 契约和状态；
6. 只有质量门通过后才原子切换 `latest`。

### P3：正式注册表和首个快照

1. 让注册表完整联合所有 JQ 数据集和 TDX；
2. 物化截至 2026-07-31 的 1m、5m、1d；
3. 生成回测投影视图；
4. 发布首个 `complete` 或 `validated_with_exclusions` 快照；
5. 对真实快照执行对象哈希、统计和抽样回读验收。

### P4：研究和回测层

1. 建设公司行为、复权因子和账户事件层；
2. 设计共享策略核心；
3. 实现 VectorBT/RQAlpha 适配器；
4. 固化成交时刻、费用、滑点、最小单位、停牌、涨跌停、T+0/T+1 等执行规则；
5. 做跨框架候选标的、方向和目标权重验收。

### P5：月度更新

1. 冻结新 TDX capture；
2. 审计父截止日之后的新数据及历史变化；
3. 新数据应用同一通用规则，历史变化默认只报告；
4. 发布后继快照，旧快照永不原地修改。

## 10. 重构时必须保留的不变量

在线模型或开发者提出设计时，不得破坏以下原则：

1. **raw 不可变**：JQ ZIP/CSV、TDX capture 和审计证据不原地修改或删除。
2. **固定截止日**：每个快照有明确 `as_of_date`，回测不能读未来数据。
3. **时点化 universe**：不能用当前上市清单回看历史；身份/生命周期裁剪先于行情修订。
4. **TDX 是条件权威，不是无条件权威**：覆盖区通过单日质量门后才优先。
5. **整日选源优先**：默认不在同一交易日混合分钟路径。
6. **小误差只标记**：不为消除差异而修改来源值。
7. **大错误不插值**：无可靠替代时隔离整日，不用日线反推分钟。
8. **排除必须显式**：批准排除有范围、原因、证据和 manifest 引用，并贯通 reader。
9. **派生单一来源**：正式 5m/1d 只来自最终 canonical 1m。
10. **未复权 canonical**：公司行为和复权在独立版本层处理。
11. **发布不可变**：对象、应用记录和 snapshot manifest 发布后不原地修改。
12. **失败不移动 latest**：只有全部质量门成功才切换指针。
13. **回测离线**：回测不直接读 raw、TDX 软件目录或网络数据。
14. **可复现**：相同 raw、配置、代码和决定必须产生相同对象和统计。

## 11. 推荐的模块边界

以下是基于当前代码的建议，不是已经实现的事实：

```text
tdx_data.sources          只负责格式解码和来源标准化
tdx_data_audit            只负责发现、分类和冻结审计证据
tdx_data_revision         负责候选、决定模型和决定验证
tdx_data.canonical        负责生命周期裁剪、日级选源和决定应用
tdx_data.derive           负责 canonical 1m -> 5m/1d
tdx_data.quality          负责候选/父快照发布门
tdx_data.snapshots        只负责通过门禁后的不可变发布与解析
tdx_data.reader           负责严格排除语义和裁剪读取
tdx_data.backtest         负责派生缓存，不成为权威数据源
```

关键接口建议以不可变标识传递，而不是传入隐式的 `latest`：

```text
audit_batch_id
tdx_capture_id
registry_id
revision_run_id
parent_snapshot_id | null
as_of_date
```
