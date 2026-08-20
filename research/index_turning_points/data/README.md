# 输入数据

`inputs/<dataset>/<version>/` 保存从 JQ 等外部环境接收的不可变输入快照。每个版本至少应包含 manifest、数据口径说明和 manifest 声明的数据文件；生成程序属于 `adapters/`，不复制进快照。

当前输入：

- [`all_a_breadth_v1_20120101_20260814`](inputs/all_a_breadth/all_a_breadth_v1_20120101_20260814/jq_breadth_export.md)：当前已验收的均线与行业宽度事实源；
- [`all_a_p1_inputs_v1_20120101_20260814`](inputs/all_a_p1_inputs/all_a_p1_inputs_v1_20120101_20260814/)：文件完整性已通过，但 MA 相等比较受浮点路径影响，只保留追溯，不进入后续研究；
- [`all_a_p1_inputs_v2_20120101_20260814`](inputs/all_a_p1_inputs/all_a_p1_inputs_v2_20120101_20260814/)：当前已验收的 P1 共用输入；完整性、内部一致性、V1 差异和使用边界见[验收报告](../docs/signals/p1_jq_input_v2_acceptance.md)。

改变股票池、日期范围、复权、行业映射、数值容差或字段口径时创建新版本，不覆盖旧目录。
