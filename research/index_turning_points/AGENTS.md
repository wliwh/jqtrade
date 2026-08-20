# 指数顶底研究协作规则

## 依赖方向

- `ground_truth/` 可以读取完整历史，只生成事后标准答案；`signals/` 的日期 `t` 只能读取 `t` 当时已知的数据。
- `evaluation/` 可以同时读取信号与事后答案，但不能反向影响信号定义、阈值或触发日期。
- `pipelines/` 只负责编排，业务逻辑放在上述领域模块；项目根目录不新增业务 `.py`。
- 外部格式放 `adapters/`。现役模块不得直接导入 `archive/`，历史数据只能经命名含版本的 legacy adapter 读取。

## 新信号放置

1. 在 `docs/signals/<signal_id>.md` 冻结方向、点时输入、原始量、阈值、缺失值和 episode 语义。
2. 新输入快照放 `data/inputs/<dataset>/<version>/`，至少包含 manifest 与口径说明；已有快照不可改写。
3. 因果信号代码放 `signals/definitions/<signal_id>.py`，外部导出程序放 `adapters/jq/`。
4. 测试按同一领域放到 `tests/index_turning_points/`；至少覆盖截断不变性和样本尾部。
5. 输出写入 `artifacts/signals/<signal_version>/` 与 `artifacts/evaluations/<evaluation_version>/`，不得覆盖既有版本。

## 验证

- 相关改动先跑 `pytest -q tests/index_turning_points`；跨模块改动跑根目录 `pytest -q`。
- 目录或文档整理还要检查 Markdown 链接、Python 语法和 `git diff --check`。
- JQ adapter 的本地测试只验证语法与确定性逻辑；真实数据覆盖和资源限制必须在 JQ 平台另行冒烟。
