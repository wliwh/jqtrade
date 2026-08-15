# JoinQuant 历史分钟数据下载

`src/` 保存可复制到 JQ 环境运行的下载脚本；`data/` 保存本地导出的参考 CSV。下载脚本自身的 `OUTPUT_DIR` 仍相对于 JQ 运行目录，避免把平台运行产物写回此仓库。

## 批量下载 ETF/LOF 1 分钟数据

只需将 `download_fund_minute_data_batches.py` 放入 JQ 研究/运行目录。
该脚本直接使用 `jqdata`，内部包含单标的分片下载和 CSV 合并逻辑，不依赖
`download_stock_minute_data.py`。

在 `download_fund_minute_data_batches.py` 顶部确认：

- `END_DATE` 是固定历史截止日；默认 `2026-04-26`，即当前 TDX 1 分钟起点的前一天。
- `BATCH_TARGET_MB` 是一批 CSV 的原始大小阈值；加入一只基金后达到或刚超过阈值即打包。
- `FQ` 固定为 `None`，不能使用前复权或后复权。
- `OUTPUT_DIR` 是 JQ 运行目录下的相对目录。

运行脚本后，它会：

1. 首次调用 `get_all_securities(['fund'], date=None)`，过滤 `type` 为 `etf`、`lof`，并把该次 universe 固化进 `state.json`。
2. 逐只调用单标的下载器；每只基金仍按 `CHUNK_DAYS` 分片请求。
3. 累计 CSV 大小达到阈值后，生成一个带 `manifest.json` 的 ZIP，并自动清理已经打包的工作 CSV。
4. ZIP 尚未删除时拒绝继续，避免在人工处理期间继续占用空间。

人工操作流程：

1. 下载 ZIP。
2. 在本地核对 ZIP 能解压，并保留其中的 `manifest.json`。
3. 只删除 JQ 上已经下载的 ZIP，保留 `OUTPUT_DIR/state.json`。
4. 再次运行脚本，自动继续下一批。

如果下载中断或单只标的失败，重新运行会从 `state.json` 中记录的标的继续。不要修改同一任务的日期、基金类型、分片大小或批次大小；如需改变口径，请换一个新的 `OUTPUT_DIR`。

ZIP 内部结构：

```text
manifest.json
data/
├── 159915_XSHE_1m_20050101_20260426.csv
└── ...
```

`manifest.json` 记录查询口径、基金类型、每个 CSV 的行数、字节数和 SHA-256，可供本地 intake 校验。
