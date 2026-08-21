# 指数顶底区域与点时信号研究

状态：`top_bottom_regions_v2`、阶段 C/D 与 P1 主候选首次冻结评测均已完成；现有候选均不足以作为独立顶底预测器，后续研究须另行预先冻结。

本项目研究严格点时信号或评分能否预测、确认指数顶底；不开发交易策略或仓位规则，也不把区域定位与信号后收益合成评测总分。

| 层 | 未来数据 | 职责 |
| --- | --- | --- |
| `ground_truth` | 可以使用 | 生成事后顶底区域、峰瓣和未来结果 |
| `signals` | 禁止使用 | 只用日期 `t` 当时已知数据生成信号与 episode |
| `evaluation` | 可以读取答案 | 分开评价区域定位与信号后 5/10/20 日 OHLC |

当前标准答案覆盖上证、沪深300、中证500、中证1000、国证2000、微盘股和全A，截止 `2026-08-14`，共 975 个区域、1494 个峰瓣。国证2000 `399303` 不等同于中证2000 `932000`。

## 导航

| 内容 | 入口 |
| --- | --- |
| 冻结评测协议 | [`top_bottom_region_evaluation_plan.md`](docs/top_bottom_region_evaluation_plan.md) |
| 候选顺序与研究备忘 | [`signal_backlog.md`](docs/signal_backlog.md) |
| 信号规格 | [`docs/signals/`](docs/signals/README.md) |
| 输入快照 | [`data/`](data/README.md) |
| 评测实现 | [`evaluation/`](evaluation/README.md) |
| 版本化结果 | [`artifacts/`](artifacts/README.md) |
| 协作与目录边界 | [`AGENTS.md`](AGENTS.md) |

## 常用命令

```bash
# 标准答案与人工审计查看器
/home/hh01/anaconda3/envs/fin/bin/python -m research.index_turning_points.pipelines.build_ground_truth --output-dir research/index_turning_points/artifacts/ground_truth/<版本>
/home/hh01/anaconda3/envs/fin/bin/python -m research.index_turning_points.pipelines.render_viewer

# 生成任一信号；<signal> 对应 pipelines/build_<signal>.py，输出目录必须为空
/home/hh01/anaconda3/bin/python -m research.index_turning_points.pipelines.build_<signal> --output-dir research/index_turning_points/artifacts/signals/<版本>

# 评测已有 signal bundle；默认读取本机 TDX vipdoc；最后运行本地研究测试
/home/hh01/anaconda3/bin/python -m research.index_turning_points.pipelines.evaluate_signal --signal-daily <signal_daily.csv> --signal-episodes <signal_episodes.csv> --ground-truth-dir research/index_turning_points/artifacts/ground_truth/index_ohlc_20260814 --evaluation-version <版本> --output-dir research/index_turning_points/artifacts/evaluations/<版本>
/home/hh01/anaconda3/bin/python -m pytest -q tests/index_turning_points
```

## 红线

- 标签参数、信号方向、阈值和评测窗口必须在查看候选成绩前冻结。
- 不用当前成分或行业回填历史，不把标签、确认日或未来结果传入信号。
- 不回填连续段最后一日，不只保留成功案例。
- 顶部/底部、预测/确认及两套评测分别报告，不合并总分。
