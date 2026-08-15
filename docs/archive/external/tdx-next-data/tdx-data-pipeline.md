# TDX ETF/LOF 三频数据管线运维说明（外部工程快照）

> 归档于 2026-08-15。本文对应另一工程；其中路径、服务和快照状态不属于 `jqtrade`。

> 更新日期：2026-08-06  
> 配置：`config/data.yaml`  
> 数据根目录：`~/.local/share/tdx-next-data`

## 1. 范围与数据源

管线处理 1,556 个 active ETF 和 428 个 LOF，共 1,984 个请求标的。

| 用途 | 来源 | 规则 |
|---|---|---|
| 首次日线 | 本地 TDX `.day` | 修正已确认的 ETF 特殊成交量编码 |
| 首次 5 分钟 | 本地 TDX `.lc5` | 只保留完整 48 根交易日 |
| 首次 1 分钟 | 本地 TDX `.lc1` | 只保留完整 240 根交易日 |
| 每日三频增量 | `tdxrs==0.6.7` | `fq=0`，保存 online raw 后校验发布 |
| ETF/LOF 分类 | `specetfdata.txt`、`speclofdata.txt` | 不以分钟文件名直接判断证券类型 |

已确认的数据边界：

- 5 分钟从 2024-10-08 开始。
- 1 分钟从 2026-04-27 开始。
- Windows 通达信与 TdxQuant 无法下载更早分钟数据，不再作为回填来源。
- 日线和分钟数据都由本机管线维护，不接受 Windows 派生文件。
- JoinQuant 等其他渠道列为早期分钟历史候选来源，但尚未接入。候选数据必须为未复权行情，并通过授权与导出条件、字段定义、交易日历、时间戳、成交量/成交额单位以及 TDX 重叠区间一致性验证。

## 2. 当前发布状态

最新快照：`snapshot-20260806T091453459470Z`，截止 2026-08-06。

| 频率 | 行数 | 对象数 | 截止时间 |
|---|---:|---:|---|
| 1d | 2,317,287 | 45 | 2026-08-06 |
| 5m | 34,697,520 | 46 | 2026-08-06 15:00 |
| 1m | 31,688,400 | 10 | 2026-08-06 15:00 |

注册表共 1,984 行：

- `available`: 1,956
- `placeholder_data`: 25
- `stale`: 3
- `missing_daily`: 0

## 3. 项目文件布局

```text
config/
├── data.yaml
├── environments/
│   ├── tdx-next.yml
│   └── tdx-next-rqalpha.yml
└── systemd/
    ├── tdx-data-update.service
    └── tdx-data-update.timer

docs/
├── tdx-next.md
└── tdx-data-pipeline.md
```

项目外数据布局：

```text
~/.local/share/tdx-next-data/
├── raw/                 本地冻结和 online 原始响应
├── objects/             内容寻址 Parquet 行情对象
├── snapshots/           不可变快照清单
├── manifests/           latest 与运行清单
├── registry/            当前注册表和不可变历史
├── validation/          校验报告
├── revisions/           历史差异与冲突事件
├── state/               同步状态和更新锁
└── tmp/                 发布前暂存
```

快照只保存对象引用。没有变化的分区不会复制，因此元数据子快照不会再次占用整套行情空间。

## 4. 配置摘要

`config/data.yaml` 的关键值：

```yaml
data_root: ~/.local/share/tdx-next-data

universe:
  expected_active_etf: 1556
  expected_lof: 428

bootstrap:
  expected_5m_bars_per_day: 48
  expected_1m_bars_per_day: 240
  exclude_incomplete_minute_days: true
  freeze_raw: true

online:
  daily_count: 80
  min5_count: 800
  min1_count: 800
  retry_count: 3

update:
  earliest_time: "16:40"
  accept_revisions: false
```

5 分钟 800 根约覆盖 16 个完整交易日；1 分钟 800 根约覆盖 3 个完整交易日。它们是增量重叠窗口，不是历史回填能力。

## 5. Bootstrap

```bash
conda activate tdx-next
tdx-data --config config/data.yaml bootstrap
```

Bootstrap 执行：

1. 锁定更新目录。
2. 扫描本地 TDX 文件和证券 catalog。
3. 冻结原始文件并记录 SHA-256。
4. 解码三频数据并应用注册表策略。
5. 排除占位数据、不完整分钟日和 stale orphan。
6. 写入内容寻址 Parquet 对象。
7. 完成单频、跨频和主键校验。
8. 原子发布 snapshot 与 latest。

## 6. 每日更新

### 6.1 一体化命令

```bash
tdx-data --config config/data.yaml update
```

正式更新在 `16:40 Asia/Shanghai` 之前会被拒绝。证券子集只允许用于 dry-run：

```bash
tdx-data --config config/data.yaml update --dry-run --symbols 510300.XSHG 159915.XSHE
```

systemd 调度使用：

```bash
tdx-data --config config/data.yaml update --skip-if-current-date
```

如果 latest 已达到当天日期，该命令返回 `already_current`，不联网、不创建 raw。

### 6.2 分步发布

```bash
tdx-data --config config/data.yaml update capture
tdx-data --config config/data.yaml update validate <update_id>
tdx-data --config config/data.yaml update publish <update_id>
```

`validate` 和 `publish` 使用同一批 raw，不重复联网。发布要求：

- capture 为完整 universe；
- 所有请求成功；
- 父快照仍是 latest；
- 新交易日分钟数据完整；
- 三频无硬冲突；
- 时间门禁通过。

### 6.3 注册表修复

旧运行若已经发布行情但漏掉注册表推进，可执行：

```bash
tdx-data --config config/data.yaml update refresh-registry <update_id>
```

该命令仅发布元数据子快照，复用所有行情对象，并校验父快照、注册表和对象 SHA-256。正常每日发布已自动推进注册表，无需额外运行。

## 7. 合并与冲突策略

Canonical 主键为：

```text
(symbol, datetime, frequency)
```

默认策略：

- 新 key 且日期晚于父快照：自动追加。
- 等价重叠：忽略并计数。
- 历史 key 数值不同：写 revision event，不覆盖。
- 父快照截止日及以前缺失的 key：标记 `historical_missing`，不自动补洞。
- `placeholder_data`：只保留 raw，不进入 canonical。
- stale 标的：分钟数据不得超出已确认日线范围；同一 capture 有真实日线和完整 48/240 根分钟线时可推进并提升状态。

冲突分类：

| 分类 | 含义 | 默认处理 |
|---|---|---|
| `rounding_only` | 严格容差内的浮点差异 | 视为等价 |
| `intraday_rebucket` | 日聚合一致、日内分桶不同 | 记录，不覆盖 |
| `aggregate_mismatch` | 完整日聚合仍不一致 | 隔离复核 |
| `incomplete_window` | 800 根窗口首日不完整 | 不做日级结论 |
| `historical_missing` | 在线存在而本地历史 key 缺失 | 进入 backfill review |

## 8. 校验与原子性

发布前检查：

- OHLC 合法性和非负成交量/成交额；
- 主键唯一；
- 5 分钟每日 48 根、1 分钟每日 240 根；
- 1d/5m/1m 完整日 OHLCV 和成交额一致性；
- raw 与对象 SHA-256；
- 请求成功数、universe 完整性和父快照一致性。

只有全部通过才更新 `manifests/latest.yaml`。失败时保留旧快照，raw 和 validation report 可用于重放和定位。

## 9. 自动调度

源文件位于 `config/systemd/`，已安装到 `~/.config/systemd/user/`。

```bash
systemctl --user status tdx-data-update.timer
systemctl --user list-timers tdx-data-update.timer
```

当前计划：工作日 16:40 运行，`Persistent=true`；失败后每 20 分钟重试，最多 3 次。首次自动触发时间为 2026-08-07 16:40 CST。

修改项目内单元后重新安装：

```bash
install -D -m 0644 config/systemd/tdx-data-update.service ~/.config/systemd/user/tdx-data-update.service
install -D -m 0644 config/systemd/tdx-data-update.timer ~/.config/systemd/user/tdx-data-update.timer
systemctl --user daemon-reload
systemctl --user restart tdx-data-update.timer
```

## 10. 常用检查

```bash
tdx-data --config config/data.yaml status
tdx-data --config config/data.yaml snapshot-list
tdx-data --config config/data.yaml snapshot-show <snapshot_id>
pytest -q
```

出现问题时依次检查：

1. `manifests/latest.yaml` 是否仍指向上一个完整快照。
2. `manifests/runs/<run_id>.yaml` 的状态。
3. `validation/<run_id>/report.yaml`。
4. `revisions/run_id=<run_id>/events.parquet`。
5. `state/sync-state.parquet` 和 `state/update.lock`。

不要手工修改已发布对象、raw、快照清单或注册表历史。需要修正时发布子快照，保留旧版本可追溯。

## 11. 后续工作

数据层下一阶段仅包括：

1. 观察每日 timer 的首次自动运行。
2. 增加周末快照完整性校验。
3. 建立公司行为事件快照和动态前复权。
4. 评估 JoinQuant 等渠道的早期未复权 5 分钟/1 分钟历史数据。

Windows 早期分钟数据回填不再列入计划。

外部分钟历史在评估通过前不得写入正式 canonical。若后续采用，应先保存独立 raw、来源授权和抓取批次信息，再进行证券映射、单位归一化、完整交易日检查和 TDX 重叠校验；任何冲突默认进入 quarantine，由新快照显式发布。
