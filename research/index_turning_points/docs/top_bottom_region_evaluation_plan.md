# 顶底区域与信号评测协议

阶段 A/B/C/D 已实现；候选状态和顺序只在 [`signal_backlog.md`](signal_backlog.md) 维护。

## 冻结参数

| 参数 | 值 |
| --- | --- |
| `label_version` | `top_bottom_regions_v2` |
| 价格带 | `min(medium_threshold × 0.20, 2%)` |
| `max_side_days` / `max_lobe_gap` | `20` / `10` 个交易日 |
| 时间分区 | 相邻中级锚点的交易日中点，不重叠 |
| 预测/确认窗口 | `5/10/20`；确认以 5 日为主 |
| `capped_confirmation_n` | `2` |

参数未读取候选成绩；修改区域参数必须升级 `label_version`，不得覆盖既有产物。

## 事后标准答案

- 中级 directional-change 点是锚点，小级点只作诊断。
- 锚点左右最多 20 日内按冻结价格带提取核心峰瓣；间隔不超过 10 日且无反向中级锚点时合并。
- 核心峰瓣用于 strict 匹配，首末峰瓣连续包络用于 loose 匹配；M/W/平台可包含多个峰瓣。
- 确认日只是事后属性，不决定区域终点或信号日期。

现役 [`ground truth`](../artifacts/ground_truth/index_ohlc_20260814/regions/top_bottom_regions_v2/) 含 975 个区域、1494 个峰瓣，其中 349 个区域为多峰。

## 点时事件

信号先保存逐日 `raw_value`、`triggered` 和数据质量，再由 [`events.py`](../signals/events.py) 派生：

| 事件 | 日期语义 |
| --- | --- |
| `onset` | 连续段首个活跃日 |
| `continuation` | 同段其余活跃日 |
| `exit` | 段后首个非活跃日 |
| `capped_confirmation(N)` | 第 N 个活跃日；短段只在看到退出时确认 |

逐日状态和已产生事件必须满足截断不变性；未退出的样本尾部保持 pending，不回填。

## 两套评测

**区域定位：** 顶部只匹配顶部、底部只匹配底部；episode 与区域一对一，额外 episode 记重复报警。匹配优先级为核心峰瓣 → 包络 → 冻结窗口 → 锚点距离。分别报告 strict/loose/window、预测/确认、单峰/多峰、指数/汇总的召回、精确率、误报、重复报警和 lead/lag。

**信号后结果：** 分 onset 与 capped confirmation 计算：

```text
terminal_return_h = close[t+h] / close[t] - 1
max_up_h          = max(high[t+1:t+h]) / close[t] - 1
max_down_h        = min(low[t+1:t+h]) / close[t] - 1
```

期限为 5/10/20 日，基线为同覆盖期完整非事件日。均值差使用期限阶 Newey–West 标准误；至少 20 个事件、30 个基线日才做推断，并报告局部与全局 BH-FDR。`close[t]` 仅是参考价，不代表可成交收益。

[`evaluate_signal.py`](../pipelines/evaluate_signal.py) 校验 signal、ground-truth 与 TDX 哈希，并写入不可覆盖的版本目录。输出结构和实现边界见 [`evaluation/README.md`](../evaluation/README.md)；区域结果与 OHLC 结果始终分开，`composite_score` 为空。
