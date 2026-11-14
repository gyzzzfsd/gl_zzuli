# 评估与优化（eval/）



## 参数

| 参数 | 作用 |
|---|---|
| `--csv` | 黄金集路径，默认 6 条样例 |
| `--mode` | `ext` 只跑确定性指标 / `ragas` 只跑 RAGAS / `all` 全跑 / `report` 只重出报告（读已有 `records.jsonl`，不跑图） |
| `--limit` | 只跑前 N 条（冒烟） |
| `--batch` | RAGAS 分批大小（默认 20，防 OOM） |
| `--judge-model` | 裁判模型；建议比生成模型强一档，留空则沿用生成模型（自评会偏高） |
| `--resume` | 断点续跑，跳过 `records.jsonl` 里已有的 qid |
| `--records` | 复用已有记录，只重算 RAGAS（省一次检索+生成的钱） |
| `--no-gate` | 只出报告、不判失败（探索阶段用） |
| `--save-baseline` | 把某次报告固化为 `eval/baseline.json` |

退出码：`0` 通过 / `1` 门禁未通过 / `2` 数据或环境问题（CI 据此区分"评测不达标"和"评测没跑起来"）。

## 文件

| 文件 | 职责 |
|---|---|
| `run_eval.py` | 命令行编排：加载 → 跑图 → 算分 → 报告 → 门禁 |
| `runner.py` | 跑 `query_app`，采 `answer`/`contexts`/耗时/报错，边跑边落盘 |
| `metrics_ext.py` | 确定性指标：数值命中、拒答正确、引用有效（零 API 成本） |
| `ragas_adapter.py` | RAGAS 五项，屏蔽 0.1/0.2 的 API 差异 |
| `report.py` | `results.csv` / `bad_cases.csv` / `summary.md` / `summary.json` |
| `gate.py` | 绝对门槛 + 相对基线双判据 |
| `dataset.py` | 黄金集加载与体检（缺列、空值、题型拼错直接报错） |
| `selfcheck.py` | 不连任何服务，验证指标与报告逻辑（CI 用） |
| `eval_doc/qa_sample.csv` | 6 条样例，用于跑通链路 |



## 优化闭环（怎么用这些数字做决策）

```
bad_cases.csv → 人工给每题打 1 个主因标签 → 统计主因分布
             → 选占比 > 20% 的主因，做单点改动（一次只改一个变量）
             → 6 条 smoke 验证方向 → 全量验证幅度 → 确认无回归才更新基线
```

| 现象 | 该改哪一层 |
|---|---|
| `context_recall` 低 + `n_contexts=0` | `item_name` 过滤过严 / 该主体没入库（先跑 `export_item_names.py --check`） |
| `context_recall` 低 + 上下文正常 | 切片粒度（`max_content_length`）/ 检索 limit / 图谱召回 |
| `context_precision` 低 | rerank 截断（`rerank_max_top_k`、`rerank_gap_abs`）、RRF 权重、内容去重 |
| `faithfulness` 低 | Prompt 未约束"不扩写"；上下文过长；数值/条件被改写 |
| 前三项正常但 `answer_correctness` 低 | 回答啰嗦、观点分散 → 收紧 prompt（2–5 句、覆盖问题对象） |
| `numeric_recall` 低 | 数值/单位丢失或改写；需要"数值只能来自证据"的硬约束 |
| `refusal_accuracy` 低 | 统一拒答话术，明确"语料无依据时必须拒答" |
| `citation_valid_rate` 低 | 引用编号越界/编造 → 校验 `【来源】` 里的 `[n]` 范围 |

## 排障

| 症状 | 原因与处理 |
|---|---|
| `上下文为空（检索未命中…）` 大面积出现 | `item_name` 与库内不一致；先 `export_item_names.py --check eval\eval_doc\qa.csv` |
| 答案全是"抱歉，我无法识别…" | 没走锁定分支：检查 `item_name_locked=True` 且 `item_names` 非空 |
| 五项指标全 NaN | 该批 `contexts` 为空，或裁判模型调用失败；看 `summary.md` 的 NaN 小节 |
| `RagasUnavailable` | 没装 ragas 或用错解释器：`.venv-eval\Scripts\python eval\run_eval.py ...` |
| 跑一半 OOM | 调小 `--batch 10` |
| 分数两次跑差很多 | 生成温度不是 0、或数据库内容变了；看 `summary.json` 的 `config_snapshot` 对比 |
