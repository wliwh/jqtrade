# 动量信号研究数据

本目录保存 JQ 导出的输入快照和本地计算结果，不作为通用程序入口。

- `momentum_index_p1_inputs_smoke_v1/`：短区间冒烟快照；只用于验证导出链路。
- `momentum_index_p1_inputs_v1/`：正式全区间 JQ 输入快照；其 `manifest.json` 是解压内容的完整性事实源，文件不得修改或补写。
- `local_results/momentum_index_p1_inputs_v1__p1_local_v2/`：从正式快照独立生成的当前本地 P1 结果，共 11 张研究表；manifest 记录输入快照哈希和各结果文件哈希。
- `local_results/` 下后续运行必须使用新的版本目录，不能覆盖已有结果。

旧 JQ P1 和一次性的 JQ/本地对照均已移出活动数据目录，见 [`../archive/`](../archive/README.md)。活动研究不读取这些结果。

任何日期、清单、代码或口径改变都必须新建数据版本。不要改写既有 manifest、CSV 或结果表来适配新分析。
