# 评估基线（2026-11-14）

第一次覆盖全部 5 台设备、35 条测试用例的完整评估结果。改动后（切片参数、Prompt、检索策略、模型版本）跑完后与此对比，判断是否真的变好。

## 采集时的系统状态

| 项 | 值 |
|---|---|
| 知识库主体 | 8 个：M1 血压计、S1 持续葡萄糖监测、N1 雾化器、G1 血糖仪、O1 血氧仪、CRISPR-Cas9、p53 肿瘤抑制蛋白、p53-MDM2 反馈回路 |
| 评估集 | `eval/eval_doc/qa.csv`，35 条（M1/S1/N1/G1/O1 各 7 条） |
| 裁判模型 | `ITEM_MODEL`/`MODEL` = qwen-flash（`backend/.env`） |
| 向量/重排 | BGE-M3（本地 `E:\app\d_app_extend\LLM_model\bge-m3`）+ bge-reranker（本地） |
| 会话 | 每条用例独立 `eval_session_<时间戳>`，`item_names=[]` 由查询图自行识别 |

## 五指标

| 指标 | 均值 | 含义 |
|---|---|---|
| faithfulness | 0.9395 | 答案是否都能被检索上下文支撑 |
| answer_relevancy | 0.8917 | 答案与问题的相关度 |
| context_precision | 0.9587 | 检索到的上下文里有用内容占比 |
| context_recall | 0.9429 | 参考答案里的事实是否都被检索到 |
| answer_correctness | 0.7495 | 答案与参考答案的事实一致度 |
| bad cases | 3 条 | `answer_correctness < 0.5` |
| 平均命中上下文 | 3.46 条 | 每条问题 rerank 后保留的文档数 |

## 3 条 bad case 的真实原因

三条的 `faithfulness / answer_relevancy / context_precision / context_recall` 全部 ≥ 0.89，只有 `answer_correctness` 低（0.326–0.457）。逐条比对后确认**都不是内容错误，而是答案比参考答案详细**：答案分条列举、补充了前置步骤或延伸说明，而参考答案只有一句骨架。

RAGAS 的 `answer_correctness` 由「事实陈述匹配 + 语义相似」合成，参考答案越短、答案越丰满，惩罚越重。所以这个指标在这份评估集上偏保守，看趋势可以，别拿单条 0.4 当"答错了"。

要让它更贴近人工判断，两个办法：
1. 把 `qa.csv` 的 answer 写成**要点清单**（把允许出现的正确陈述尽量列全）；
2. 用 `eval/metrics_ext.py` 里的关键信息命中率（自己算，不受长度影响）作为补充指标。

## 分设备明细（同一批 35 条按产品拆分）

| 设备 | 条数 | faithfulness | answer_relevancy | context_precision | context_recall | answer_correctness | 平均上下文 |
|---|---|---|---|---|---|---|---|
| M1 血压计 | 7 | 0.943 | 0.943 | 0.948 | 0.857 | 0.618 | 3.86 |
| S1 持续葡萄糖监测 | 7 | 0.982 | 0.668 | 0.917 | 0.857 | 0.746 | 3.29 |
| N1 雾化器 | 7 | 0.865 | 0.950 | 1.000 | 1.000 | 0.771 | 3.14 |
| G1 血糖仪 | 7 | 0.916 | 0.941 | 0.952 | 1.000 | 0.837 | 3.29 |
| O1 血氧仪 | 7 | 0.991 | 0.956 | 0.976 | 1.000 | 0.775 | 3.71 |





