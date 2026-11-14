"""知识库 RAG 评估（RAGAS 五项指标）。


    step_1 初始化 LLM / Embedding / Metrics
    step_2 从 eval/eval_doc/qa.csv 读取 question + answer
    step_3 跑本项目查询图，采集 answer 与 contexts
    step_4 组装成 RAGAS Dataset
    step_5 调用 ragas.evaluate() 计算五项指标
    step_6 导出 eval/eval_results.csv 与 eval/bad_cases.csv

用法（在仓库根目录或任意目录执行均可）：
    python eval/ragas_eval.py

前置条件：
    1. pip install -r requirements-eval.txt
    2. .env 里有可用的 OPENAI_API_KEY / OPENAI_API_BASE（裁判模型）
    3. Milvus、Neo4j、MongoDB 可连（查询图依赖它们）
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# 依赖缺失时给出安装提示，而不是一屏 traceback
try:
    import pandas as pd
    # Dataset 是 HuggingFace datasets 的，不是 ragas 的（ragas 0.2 顶层没有 Dataset）
    from datasets import Dataset
    from langchain_core.embeddings import Embeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
except ImportError as exc:
    print(f"缺少评估依赖：{type(exc).__name__}: {exc}")
    print("请先执行：pip install -r requirements-eval.txt")
    raise SystemExit(2)

# 仓库根目录与 backend 都要在 sys.path 上，才能导入 app 包
REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from app.processor.query_process.main_graph import query_app
    from app.utils.bge_m3_embedding_util import generate_hybrid_embeddings, get_beg_m3_embedding_model
    from app.utils.llm_client_util import get_llm_client
except ImportError as exc:
    print(f"导入项目模块失败：{type(exc).__name__}: {exc}")
    print("请用装好 backend/requirements.txt 的解释器运行（例如项目的 venv）。")
    raise SystemExit(2)

# 只跑前 N 条，用于先验证链路（0 表示全部）。
# 可用环境变量覆盖：$env:EVAL_QA_LIMIT=5 无需改代码；整份评估很吃内存和时间，建议先小后大
EVAL_QA_LIMIT = int(os.getenv("EVAL_QA_LIMIT", "0"))

# answer_correctness 低于该值、或该指标为 NaN，即判为 bad case
BAD_CASE_THRESHOLD = 0.5

QA_CSV_PATH = Path(__file__).resolve().parent / "eval_doc" / "qa.csv"


class ProjectBGEEmbeddings(Embeddings):
    """把项目内的 BGE-M3 适配成 LangChain Embeddings。

    只暴露 dense：RAGAS 不负责检索，只拿最终 contexts 文本算语义相似度，
    sparse 在这里用不上。
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成稠密向量。

        Args:
            texts: 待编码文本列表。

        Returns:
            向量列表；模型加载失败时返回空列表（让 RAGAS 记 NaN，而不是整轮崩掉）。
        """
        # 1. 空输入直接返回，避免无意义的模型加载
        if not texts:
            return []

        # 2. 加载模型（失败返回 None）
        model = get_beg_m3_embedding_model()
        if model is None:
            print("[warn] BGE-M3 模型加载失败，相关指标将记为 NaN")
            return []

        # 3. 复用项目封装（同时产出 dense/sparse，这里只取 dense）
        vectors = generate_hybrid_embeddings(model, list(texts)) or {}
        return [list(map(float, vector)) for vector in vectors.get("dense") or []]

    def embed_query(self, text: str) -> List[float]:
        """编码单条查询。

        Args:
            text: 查询文本。

        Returns:
            稠密向量；失败时返回空列表。
        """
        dense = self.embed_documents([text])
        return dense[0] if dense else []


def strip_image_suffix_for_eval(answer: str) -> str:
    """去掉答案末尾的【图片】区块。

    评估比对的是文字答案，图片 URL 会干扰 answer_relevancy / correctness 的判断。

    Args:
        answer: 查询图输出的原始答案。

    Returns:
        只保留正文的答案。
    """
    return (answer or "").split("【图片】", 1)[0].strip()


def step_1_init_eval_components():
    """初始化评估组件（LLM、Embedding、Metrics）。

    Returns:
        (ragas_llm, ragas_embeddings, ragas_metrics)
    """
    # 1. 裁判 LLM：复用项目客户端（默认 qwen-flash，可用 ITEM_MODEL 覆盖）
    llm = LangchainLLMWrapper(get_llm_client())

    # 2. 语义指标用的向量模型：项目自己的 BGE-M3
    embeddings = LangchainEmbeddingsWrapper(ProjectBGEEmbeddings())

    # 3. 五项指标：检索侧两项、生成侧两项、参考答案对齐一项
    metrics = [
        faithfulness,       # 忠实度：答案是否基于检索到的上下文
        answer_relevancy,   # 答案相关性：答案是否回答了问题
        context_precision,  # 上下文精确度：相关切片在结果中的排名情况
        context_recall,     # 上下文召回率：是否检索到了回答所需的信息
        answer_correctness,  # 答案正确性：答案与参考答案的匹配程度
    ]

    return llm, embeddings, metrics


def step_2_load_test_samples_from_csv(csv_path: str) -> List[Dict[str, str]]:
    """从 CSV 加载测试样本（列名要求：question, answer）。

    Args:
        csv_path: qa.csv 路径（两列：question、answer；answer 即参考答案）。

    Returns:
        [{"question": str, "ground_truth": str}]
    """
    # 1. 文件存在性
    if not os.path.exists(csv_path):
        print(f"未找到测试数据文件: {csv_path}")
        return []

    # 2. 读取（utf-8-sig 兼容 Excel 另存的 BOM）
    print(f"正在从 CSV 加载测试数据: {csv_path}")
    try:
        df_qa = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001 - 读取失败直接返回空，由调用方判断
        print(f"读取 CSV 失败: {exc}")
        return []

    # 3. 逐行映射：CSV 的 answer 列在 RAGAS 里叫 ground_truth
    test_samples: List[Dict[str, str]] = []
    for _, row in df_qa.iterrows():
        test_samples.append({
            "question": str(row["question"]).strip(),
            "ground_truth": str(row["answer"]).strip(),
        })

    # 4. 可选截断，便于先用少量样本验证链路
    if EVAL_QA_LIMIT and len(test_samples) > EVAL_QA_LIMIT:
        test_samples = test_samples[:EVAL_QA_LIMIT]
        print(f"已按 EVAL_QA_LIMIT 截断为前 {len(test_samples)} 条")

    print(f"成功加载 {len(test_samples)} 条测试用例。")
    return test_samples


def step_3_run_pipeline(questions: List[str]) -> Dict[str, List[Any]]:
    """运行现有 RAG Pipeline，采集 answer 与 contexts。

    Args:
        questions: 问题列表。

    Returns:
        {"answer": [str], "contexts": [list[str]]}
    """
    answers: List[str] = []
    contexts: List[List[str]] = []

    print(f"开始运行 Pipeline 推理，共 {len(questions)} 个问题...")

    # 1. 每次评估只用一个会话（带时间戳），避免多次评估的历史互相污染
    session_id = f"eval_session_{datetime.now():%Y%m%d%H%M%S}"

    for index, question in enumerate(questions, start=1):
        # 2. 构造输入状态：评估是单轮、非流式；item_names 留空由查询图自行识别主体
        #    task_id 必须给：answer_output_node 会下标取它做任务态/流式上报，缺了直接 KeyError
        input_state = {
            "original_query": question,
            "session_id": session_id,
            "task_id": f"eval_task_{index}",
            "is_stream": False,
            "history": [],
            "item_names": [],
        }

        try:
            # 3. 调用 langgraph（同步模式，答案落在 state["answer"]）
            result_state = query_app.invoke(input_state)

            # 4. 取答案并剥掉【图片】区块
            answer = strip_image_suffix_for_eval(result_state.get("answer", ""))
            answers.append(answer)

            # 5. 取重排后的切片作为 contexts（RAGAS 要求 list[str]）
            #    注意：本项目 reranked_docs 的内容字段是 content，不是 text
            reranked_docs = result_state.get("reranked_docs", [])
            doc_texts = []
            for doc in reranked_docs or []:
                text = (doc.get("content") or "").strip()
                if text:
                    doc_texts.append(text)
            contexts.append(doc_texts)

            print(f"  [{index}/{len(questions)}] 上下文 {len(doc_texts)} 条 :: {question[:32]}")

            # 主体识别失败时查询图会直接返回拒绝话术，评估分数会全低，
            # 这里单独提示，避免把"没跑到检索"误判成"检索质量差"
            if "无法识别" in answer:
                print(f"  [warn] 该问题未识别出知识主体，答案被短路：{answer[:40]}")
        except Exception as exc:  # noqa: BLE001 - 单条失败不能打断整轮
            print(f"Error processing question '{question}': {exc}")
            answers.append("Error generating response")
            contexts.append([])

    return {"answer": answers, "contexts": contexts}


def step_4_build_dataset(test_data: List[Dict[str, str]], pipeline_results: Dict[str, List[Any]]) -> Dataset:
    """构建 RAGAS 数据集。

    Args:
        test_data: step_2 的样本列表。
        pipeline_results: step_3 的推理结果。

    Returns:
        ragas.Dataset
    """
    questions = [item["question"] for item in test_data]

    # 1. schema 里参考答案是 str，所以这里是 list[str]，不能是 list[list[str]]
    ground_truths = [item["ground_truth"] for item in test_data]

    data_dict = {
        "question": questions,
        "answer": pipeline_results["answer"],
        "contexts": pipeline_results["contexts"],
        "ground_truth": ground_truths,
    }

    # 2. 列名沿用 ragas 0.1 的 ground_truth（即 CSV 的 answer 列）。
    #    实测 ragas 0.2.15 会自动把它映射为内部字段 reference：
    #    evaluate() 输出 to_pandas() 的列名会变成 user_input / response / retrieved_contexts / reference，
    #    指标列名不变，所以 step_6 里的 answer_correctness 仍可直接用
    return Dataset.from_dict(data_dict)


def step_5_run_ragas_evaluation(dataset: Dataset, metrics, llm, embeddings):
    """执行 RAGAS 评估。

    Args:
        dataset: step_4 构建的数据集。
        metrics: 指标列表。
        llm: 裁判 LLM。
        embeddings: 向量模型。

    Returns:
        ragas EvaluationResult。
    """
    print("开始 Ragas 评估...")

    # raise_exceptions=False：单条样本某个指标失败（上下文为空、解析失败）时，
    # 该格记 NaN 而不是让整次评估中断
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )


def step_6_export_results_and_bad_cases(results, metrics, bad_case_threshold: float = 0.5):
    """导出逐样本结果与 bad cases。

    Args:
        results: step_5 的返回值。
        metrics: 指标列表（用 m.name 取列名）。
        bad_case_threshold: answer_correctness 阈值，低于它或为 NaN 即 bad case。

    Returns:
        原始 results 对象。
    """
    print("\n========== 评估结果 ==========")
    print(results)

    df = results.to_pandas()
    metric_names = [m.name for m in metrics]

    # 1. 末尾追加一行"平均值"，便于直接在 CSV 里看整体表现
    avg_row = {col: "" for col in df.columns}
    if "question" in df.columns:
        avg_row["question"] = "平均值"
    elif "user_input" in df.columns:
        avg_row["user_input"] = "平均值"
    elif len(df.columns) > 0:
        avg_row[df.columns[0]] = "平均值"

    for metric in metric_names:
        if metric in df.columns:
            avg_row[metric] = pd.to_numeric(df[metric], errors="coerce").mean()

    df_to_export = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # 2. 导出明细（utf-8-sig 让 Excel 直接打开不乱码）
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "eval_results.csv")
    df_to_export.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n详细结果已保存至: {output_file}")

    # 3. 提取 bad cases：answer_correctness 低于阈值，或该指标为 NaN
    bad_case_metric = "answer_correctness"
    print(f"\n正在提取 Bad Cases ({bad_case_metric} 阈值: {bad_case_threshold})...")

    if bad_case_metric not in df.columns:
        print(f"[warn] 结果里没有 {bad_case_metric} 列，跳过 bad case 提取")
        return results

    correctness = pd.to_numeric(df[bad_case_metric], errors="coerce")
    bad_cases_df = df[correctness.lt(bad_case_threshold) | correctness.isna()]

    if not bad_cases_df.empty:
        bad_cases_file = os.path.join(output_dir, "bad_cases.csv")
        bad_cases_df.to_csv(bad_cases_file, index=False, encoding="utf-8-sig")
        print(f"Bad cases 已保存至: {bad_cases_file}（{len(bad_cases_df)} 条）")
    else:
        print(f"恭喜！没有发现 {bad_case_metric} 低于阈值的 Bad Cases。")

    # 4. 终端里再打一张五项均值表，省得去翻 CSV
    print("\n========== 五项指标均值 ==========")
    for metric in metric_names:
        if metric in df.columns:
            value = pd.to_numeric(df[metric], errors="coerce").mean()
            print(f"  {metric:<20}{value:.4f}" if pd.notna(value) else f"  {metric:<20}NaN")

    return results


if __name__ == "__main__":
    # --- step_1: 初始化评估组件 ---
    ragas_llm, ragas_embeddings, ragas_metrics = step_1_init_eval_components()

    # --- step_2: 加载测试数据 ---
    ragas_test_samples = step_2_load_test_samples_from_csv(str(QA_CSV_PATH))
    if not ragas_test_samples:
        print("没有可用测试样本，评估终止。")
        raise SystemExit(2)

    # --- step_3: 运行 RAG pipeline ---
    step_3_questions = [item["question"] for item in ragas_test_samples]
    step_3_pipeline_results = step_3_run_pipeline(step_3_questions)

    # --- step_4: 构建评估数据集 ---
    ragas_dataset = step_4_build_dataset(ragas_test_samples, step_3_pipeline_results)

    # --- step_5: 执行 Ragas 评估 ---
    ragas_results = step_5_run_ragas_evaluation(
        dataset=ragas_dataset,
        metrics=ragas_metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    # --- step_6: 导出结果并提取 bad cases ---
    step_6_export_results_and_bad_cases(
        results=ragas_results,
        metrics=ragas_metrics,
        bad_case_threshold=BAD_CASE_THRESHOLD,
    )