# 全A MA20 宽度—指数背离 V1

状态：阶段 E 冻结定义；定义先于首次评测落盘，评测后未修改。用于检验全A指数仍接近阶段高点、但参与上涨的股票比例已明显脱离自身阶段高点时，是否具有顶部提示性。

## 输入与点时边界

- 宽度使用已验收 `all_a_p1_inputs_v2_20120101_20260814` 的全A `breadth_ma20`；日期 `t` 的点时股票池、复权价格和宽度均在当日收盘后可得。
- 指数价格使用本机 TDX `ds/lday/62#000985.day` 的全A指数收盘价，只读取不晚于 `t` 的记录；pipeline 保存源文件哈希，不读取 ground-truth 标签或未来结果。
- 两个阶段高点都使用包含当日的过去 60 个宽度交易日，约为一个季度；前 59 日只保留连续量，不触发。
- TDX 在宽度日历内缺少 2017-04-10、2017-06-19 两日。缺价日不触发，不插值、不删宽度日；滚动窗口至少需要 59 个有效指数收盘价，两个缺口同时位于窗口时保持不可用。缺失日期和逐日有效数进入 manifest 与 daily 审计字段。

## 冻结公式

```text
price_high_t   = rolling_max(all_a_close, 60)
breadth_high_t = rolling_max(breadth_ma20, 60)

price_distance_t   = 1 - all_a_close_t / price_high_t
breadth_distance_t = 1 - breadth_ma20_t / breadth_high_t
divergence_t       = breadth_distance_t - price_distance_t

trigger_t = price_distance_t <= 0.02 and divergence_t >= 0.20
```

- 初始方向固定为 `top`；价格必须距离60日高点不超过2%，同时宽度相对高点的退潮比价格至少深20个百分点。
- `raw_value` 为 `divergence_t`；输出保留指数/宽度高点、两种距离、MA20计数和数据可用状态。
- `universe_size` 使用点时全A成分数，`valid_count` 使用当日 MA20 有效股票数。
- 信号内样本量预检得到531个触发日、140个episode，只用于确认能进入冻结推断，没有读取顶底区域或指数未来收益。

## Episode 与评测

连续触发日使用统一 [`events.py`](../../signals/events.py) 合并；onset 为首个活跃日，capped confirmation 固定为第2个活跃日，单日短段在退出日确认，样本尾部不回填。

该信号只按顶部方向进入固定区域定位与5/10/20日OHLC评测。窗口、阈值、缺价策略和方向在首次评测后保持冻结；任何变更必须升级版本，不能根据本样本成绩重新搜索窗口或阈值。

## 结果入口

正式产物见 [`signal bundle`](../../artifacts/signals/breadth_price_divergence_v1_20120104_20260814/) 和 [`stage_d_v1`](../../artifacts/evaluations/breadth_price_divergence_v1_20120104_20260814__stage_d_v1/)；跨信号结论见 [`signal_backlog.md`](../signal_backlog.md#首次冻结评测总览)。
