# 指数 P1 本地复验 V1

## 结论

正式 JQ 输入快照已经通过 manifest 完整性检查，本地 P1 重算复现了封存 JQ 结果的研究结论。每日 Rank IC、协议门槛和覆盖统计均一致；可观察的数值差异只存在于 Pearson IC 及其派生汇总，绝对差最大约 `3.1e-8`，不改变符号、显著性判断或任何协议门槛。

因此，后续指数参数诊断可以在本地完成，不必在 JQ 重复运行耗时的 `evaluate_primary_groups`。JQ 只负责生成版本化原始输入；宽基成分股验证应另建输入版本和研究协议。

## 输入身份

- 数据集：`momentum_index_p1_inputs_v1`
- JQ 生成时间：2026-08-22 13:44:22
- JQ 区间：2015-06-05 至 2026-08-20
- 正式研究区间：2016-01-01 至 2026-08-20
- 输入 manifest SHA-256：`42cc3582448fde506b547c556d6c962c8c58e4a26b2ecb00b439b7a2c46aa923`
- JQ exporter logic hash：`54e87cebcb6193767ffb7cd50872d0bd0c18dadabc475abf6f9aadeddb7ed309`
- JQ 环境：Python 3.6.7、pandas 0.23.4、numpy 1.14.6
- 解压目录：`data/momentum_index_p1_inputs_v1/`

解压目录共 7 个文件、148,858 行 CSV，所有文件的字节数、列名、行数和 SHA-256 均与 JQ manifest 一致。项目工作区未保留正式 tar 本体，因此这里能证明解压内容未偏离 manifest，但不能重新计算 tar 文件整体的 SHA-256；原始 tar 应在下载归档中继续保留。

### 冒烟顺序偏离

manifest 记录的正式包生成时间是 13:44:22，冒烟包是 13:45:03；正式包的 exporter logic hash 为 `54e87c...`，冒烟包为 `d92837...`。这说明本次执行不是严格的“同一加载代码身份先冒烟、再正式”，可能与 Notebook 定义单元的执行顺序或重新定义有关，现有证据不足以进一步归因。

该偏离不否定正式输入：正式目录自身通过全部 payload 校验，并且其本地 Rank IC 与早先独立运行的 JQ P1 逐日完全一致。不过，下一次新建数据版本时必须先重启内核、执行一次完整定义单元，确认冒烟和正式 manifest 的 exporter logic hash 相同后再接受正式包。

## 数据覆盖

| 项目 | 正式快照 |
| --- | ---: |
| 交易日 | 2,725 |
| 宽基价格行 | 27,250 |
| 风格价格行 | 32,700 |
| 申万一级价格行 | 86,063 |
| 宽基有效指数 | 10 / 10 |
| 历史申万一级有效指数 | 32 / 38 |
| 风格有效指数 | 12 / 12 |

6 个无有效记录的申万代码属于旧分类：`801060`、`801070`、`801090`、`801100`、`801190`、`801220`。本地加载器仍按 catalog 重建这 6 个全空列，因而不会因为长表没有记录而静默改变冻结池。

## 本地运行身份

- 运行入口：`local/run_p1_from_snapshot.py`
- P1 源码 SHA-256：`f10d2f8b3d72ac4bfbda42cb4d1280b495b6fc36a6d0452e54516ae5fdca620c`
- 本地环境：Python 3.9.0、pandas 2.2.3、numpy 2.0.2
- 运行耗时：约 2 分 23 秒
- 输出目录：`data/local_results/momentum_index_p1_inputs_v1__p1_local_v1/`
- 输出 manifest SHA-256：`82bb4afed05c4bdf62af762e8115365c045c18c56870232486090dc49f7a2bb6`

输出 manifest 所列 13 张结果表均通过字节数与 SHA-256 复核。`runtime_environment` 额外记录输入 dataset、输入 manifest hash、exporter logic hash 和解压目录，避免结果脱离输入快照。

## 与封存 JQ P1 的对照

参考结果是 `archive/p1_jq_multi_universe_v2_20260822/data/` 中的 13 张 JQ 表。

| 对照项 | 结果 |
| --- | --- |
| 表集合 | 13 / 13 相同 |
| 表行数与字段集合 | 全部相同 |
| 每日 Rank IC | 38,745 行逐值完全相同，最大绝对差为 0 |
| 覆盖统计 | 按 `universe_group/code` 排序后完全相同 |
| 协议与协议门槛 | 完全相同 |
| 分组、Top-K、R² 双排序、年度主 IC | 仅机器浮点舍入，最大绝对差小于 `9e-16` |
| 每日 Pearson IC | 最大绝对差 `2.3154e-9` |
| IC 汇总 | 最大绝对差 `3.0209e-8`；在 `rtol=atol=1e-7` 下全部一致 |
| 字段顺序 | 若干汇总表顺序不同，来自 pandas 0.23.4 与 2.2.3 的拼接顺序差异；字段集合和数值一致 |
| 运行环境表 | 按设计不同，并增加本地输入 provenance |

Pearson 及汇总的微小差异与 JQ 原始浮点数先按 `%.12g` 写入 CSV、再由不同 pandas/numpy 版本读取计算相符。Rank IC 依赖排序而不是极小的数值尾差，因此得到逐日精确复现。

## 冻结主单元复验

主单元仍为 `slope_x_r2 / L=25 / H=5`。

| 截面 | 开发期 Rank IC | 验证期 Rank IC | 锁定样本外 Rank IC | 五项门槛 |
| --- | ---: | ---: | ---: | --- |
| 宽基 | 0.088112 | 0.020686 | 0.025305 | 3 / 5；分组差与 Top1 超额失败 |
| 申万一级 | 0.026167 | -0.021016 | -0.003460 | 2 / 5；验证期 IC、平台和分组差失败 |
| 风格 | 0.067227 | 0.076077 | 0.037294 | 5 / 5，通过 |

这再次确认：当前证据最扎实的是风格指数动量；宽基有正 Rank IC，但不足以支持直接选择 Top1；行业动量在验证期没有通过冻结协议。

## 复现命令

先验证解压目录：

```bash
python -m research.momentum_signal_validation.local.validate_p1_index_archive \
  research/momentum_signal_validation/data/momentum_index_p1_inputs_v1
```

再写入一个全新的版本化输出目录：

```bash
python -m research.momentum_signal_validation.local.run_p1_from_snapshot \
  research/momentum_signal_validation/data/momentum_index_p1_inputs_v1 \
  --output research/momentum_signal_validation/data/local_results/momentum_index_p1_inputs_v1__p1_local_v2
```

现有 `p1_local_v1` 不得覆盖；重跑必须使用新版本名。
