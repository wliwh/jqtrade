# 本地 ETF/LOF 回测系统

> 更新日期：2026-08-06  
> 状态：三频数据 Bootstrap、首次全量增量发布和每日定时更新均已完成

## 1. 当前结论

- 回测数据放在项目外的 `~/.local/share/tdx-next-data`。
- 请求 universe 固定为 1,984 个标的：1,556 个 active ETF 和 428 个 LOF。
- 当前注册表状态：1,956 个 `available`、25 个 `placeholder_data`、3 个 `stale`，`missing_daily=0`。
- 数据频率为未复权日线、原生 5 分钟和原生 1 分钟，统一使用 `fq=0`。
- 最新快照为 `snapshot-20260806T091453459470Z`，截止 2026-08-06。
- 回测只读取不可变快照，不直接读取 TDX `vipdoc`，也不在回测期间联网。
- Windows 通达信和 TdxQuant 已验证无法取得当前本地范围之前的分钟数据。因此不再保留 Windows 早期分钟回填项目，现有分钟起点作为已知数据边界。

当前数据范围：

| 频率 | 行数 | 时间范围 | 对象数 |
|---|---:|---|---:|
| 日线 | 2,317,287 | 截止 2026-08-06 | 45 |
| 5 分钟 | 34,697,520 | 2024-10-08 09:35 至 2026-08-06 15:00 | 46 |
| 1 分钟 | 31,688,400 | 2026-04-27 09:31 至 2026-08-06 15:00 | 10 |

## 2. 技术路线

```text
本地 TDX .day/.lc5/.lc1 ── Bootstrap ─┐
                                      ├─ Canonical Parquet ─ Snapshot ─ 回测
tdxrs fq=0 ── 每日三频增量 ────────────┘
```

- 本地 TDX 文件负责首次历史导入。
- `tdxrs==0.6.7` 负责每日重叠窗口和新增数据。
- raw、标准化数据、校验报告、修订事件和快照清单分层保存。
- Parquet 对象按内容寻址；新快照复用未变化对象，不整库复制。
- 所有发布先校验，最后原子切换 `manifests/latest.yaml`。

## 3. 环境与配置

正式环境位于 `/home/ai0/anaconda3/envs/tdx-next`，不安装在项目目录。

```bash
conda env create -f config/environments/tdx-next.yml
conda activate tdx-next
pip install -e .
```

RQAlpha 使用隔离环境：

```bash
conda env create -f config/environments/tdx-next-rqalpha.yml
```

主要文件：

```text
config/data.yaml                         数据管线配置
config/environments/tdx-next.yml        数据与 VectorBT 环境
config/environments/tdx-next-rqalpha.yml RQAlpha 隔离环境
config/systemd/                         每日更新单元源文件
docs/tdx-data-pipeline.md               数据运维说明
```

## 4. 已完成能力

- TDX `.day`、`.lc5`、`.lc1` 解码。
- ETF 特殊成交量编码修正。
- ETF/LOF catalog、请求注册表和数据状态管理。
- immutable raw freeze、CanonicalBar、内容寻址对象和 snapshot manifest。
- 日线、5 分钟、1 分钟 Bootstrap 与完整日校验。
- `tdxrs` 三频增量、请求重试、重叠合并和原子发布。
- 历史冲突分类：`rounding_only`、`intraday_rebucket`、`aggregate_mismatch`、`incomplete_window`、`historical_missing`。
- 跨频 OHLCV/成交额校验。
- 注册表覆盖范围随正式发布推进；完整真实三频可将 `stale` 提升为 `available`。
- systemd user timer：工作日 16:40 自动运行，同日已发布时不联网。

首次正式全量增量运行：

```text
update_id: update-20260806T084804693607Z
标的数: 1,984
成功请求: 5,952 / 5,952
新增 1d: 1,959
新增 5m: 93,888
新增 1m: 469,440
硬跨频冲突: 0
```

行情快照 `snapshot-20260806T090445001090Z` 发布后，又发布了只修正注册表的子快照 `snapshot-20260806T091453459470Z`。该子快照新增行情对象数为 0，并将 `501186.XSHG` 提升为 `available`。

## 5. 回测设计

采用两阶段方案：

1. VectorBT：批量参数粗筛、敏感性分析和候选组合生成。
2. RQAlpha：手续费、滑点、订单约束和账户行为精确验证。

策略核心应与框架解耦：输入行情和参数，输出目标权重或订单意图。交易使用 raw 价格；复权只用于信号计算，现金分红单独计入账户，避免收益重复计算。

在开始大规模参数寻优前，还需完成：

1. 公司行为事件快照和动态前复权。
2. RQAlpha 单 ETF 买入持有验收。
3. 迁移一份最简单的 JoinQuant ETF 策略。
4. 接入已有参数生成、去重和任务记录能力。

另行评估 JoinQuant 等渠道能否提供早于现有起点的未复权 5 分钟和 1 分钟数据。评估通过后也应先进入独立 raw intake 和 quarantine，不直接覆盖现有 TDX canonical。

## 6. 已知边界

- 5 分钟历史起点为 2024-10-08。
- 1 分钟历史起点为 2026-04-27。
- TDX 在线窗口主要用于增量和重叠校验，不能替代更早分钟历史。
- Windows TDX/TdxQuant 无法提供更早分钟数据，该方向已关闭。
- JoinQuant 等其他合法渠道可作为候选来源，但必须先验证未复权口径、授权与导出能力、字段和时间戳定义，以及与现有 TDX 重叠区间的一致性。目前仅登记评估，不进入正式管线。
- 历史重叠冲突默认只记录，不自动覆盖本地 canonical。
- 25 个零成交占位 LOF 保留在注册表和 raw 层，不进入 canonical bars。

