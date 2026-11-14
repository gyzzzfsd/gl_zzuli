"""RAGAS 五项指标适配层。

为什么单独一层：RAGAS 的 API 在 0.1 → 0.2 之间变了（`ground_truth` 改 `reference`、
部分指标从实例变成类、wrapper 改名），把版本差异全部关在这个文件里，
其他模块只认"records 进、分数出"。

安装（与主依赖隔离，避免和 langgraph 的版本约束打架）：
    pip install -r requirements-eval.txt
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from eval._bootstrap import ensure_backend_on_path
from eval.config import EvalConfig

# 参与评估的五项指标，顺序即报告里的顺序
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)


class RagasUnavailable(RuntimeError):
    """RAGAS 未安装或版本不兼容时抛出，提示如何修。"""


def _import_ragas():
    """导入 ragas 与 evaluate。

    Returns:
        (ragas 模块, evaluate 函数)。

    Raises:
        RagasUnavailable: 未安装或导入失败。
    """
    try:
        import ragas  # noqa: F401
        from ragas import evaluate
    except Exception as exc:  # noqa: BLE001 - 各种 ImportError/版本错误统一转成可读提示
        raise RagasUnavailable(
            "未安装或无法导入 ragas。请执行：pip install -r requirements-eval.txt"
            f"（原始错误：{type(exc).__name__}: {exc}）"
        ) from exc
    return ragas, evaluate


def _build_dataset(batch: Sequence[Dict[str, Any]]):
    """把记录批转成 RAGAS Dataset。

    兼容两种列名：0.2 用 `reference`，0.1 用 `ground_truth`。
    空上下文补成 [""]，让指标算成低分/NaN，而不是整批抛异常。

    Args:
        batch: 记录列表。

    Returns:
        ragas.Dataset 实例。
    """
    from ragas import Dataset

    # 1. 组装列
    payload: Dict[str, List[Any]] = {
        "question": [r.get("question", "") for r in batch],
        "answer": [r.get("answer", "") or "" for r in batch],
        "contexts": [(r.get("contexts") or [""]) for r in batch],
        "reference": [r.get("reference", "") for r in batch],
        # 额外列用于把分数映射回 qid；RAGAS 会原样保留在 to_pandas() 里
        "qid": [r.get("qid", "") for r in batch],
    }

    # 2. 先按 0.2 的列名建，失败再退回 0.1 的 ground_truth
    try:
        return Dataset.from_dict(payload)
    except Exception:
        payload["ground_truth"] = payload.pop("reference")
        return Dataset.from_dict(payload)


def _build_metrics() -> List[Any]:
    """构建指标列表，兼容"实例"与"类"两种形态。

    Returns:
        指标对象列表。

    Raises:
        RagasUnavailable: 一个指标都取不到（版本差异过大）。
    """
    try:
        from ragas import metrics as ragas_metrics
    except Exception as exc:  # noqa: BLE001
        raise RagasUnavailable(f"ragas.metrics 导入失败：{exc}") from exc

    picked: List[Any] = []
    missing: List[str] = []
    for name in METRIC_NAMES:
        obj = getattr(ragas_metrics, name, None)
        if obj is None:
            missing.append(name)
            continue
        # 0.2 起部分指标是类，需要实例化；实例化的指标直接用
        if isinstance(obj, type):
            try:
                obj = obj()
            except Exception:  # noqa: BLE001 - 需要构造参数的指标就跳过
                missing.append(name)
                continue
        picked.append(obj)

    if not picked:
        raise RagasUnavailable(f"ragas 里取不到任何指标，缺失：{missing}")
    if missing:
        print(f"[warn] 以下指标在当前 ragas 版本不可用，已跳过：{missing}")
    return picked


def _build_llm(cfg: EvalConfig):
    """构建裁判 LLM（LangChain wrapper）。

    为什么默认建议单独指定裁判模型：用生成模型自评会系统性偏高。

    Args:
        cfg: 评估配置。

    Returns:
        LangchainLLMWrapper 实例。

    Raises:
        RagasUnavailable: 客户端创建失败或 wrapper 不可用。
    """
    ensure_backend_on_path()
    try:
        from ragas.llms import LangchainLLMWrapper
    except Exception as exc:  # noqa: BLE001
        raise RagasUnavailable(f"ragas.llms 导入失败：{exc}") from exc

    from app.utils.llm_client_util import get_llm_client

    client = get_llm_client(cfg.judge_model or None)
    if client is None:
        raise RagasUnavailable("裁判 LLM 初始化失败：检查 OPENAI_API_KEY / OPENAI_API_BASE / ITEM_MODEL")
    return LangchainLLMWrapper(client)


def _build_embeddings():
    """把项目内的 BGE-M3 适配成 LangChain Embeddings。

    Returns:
        LangchainEmbeddingsWrapper 实例。
    """
    ensure_backend_on_path()
    try:
        from langchain_core.embeddings import Embeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except Exception as exc:  # noqa: BLE001
        raise RagasUnavailable(f"embeddings wrapper 不可用：{exc}") from exc

    from app.utils.bge_m3_embedding_util import (
        generate_hybrid_embeddings,
        get_beg_m3_embedding_model,
    )

    class ProjectBGEEmbeddings(Embeddings):
        """只暴露 dense 的 BGE-M3 适配器（RAGAS 不需要 sparse）。"""

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            # 1. 模型加载失败返回空列表，让上层记录 NaN，而不是在这里崩
            model = get_beg_m3_embedding_model()
            if model is None or not texts:
                return []

            # 2. 复用项目封装（失败返回 None）
            vectors = generate_hybrid_embeddings(model, list(texts)) or {}
            dense = vectors.get("dense") or []

            # 3. 统一成 list[float]，避免 numpy 标量影响序列化
            return [list(map(float, v)) for v in dense]

        def embed_query(self, text: str) -> List[float]:
            dense = self.embed_documents([text])
            return dense[0] if dense else []

    return LangchainEmbeddingsWrapper(ProjectBGEEmbeddings())


def _batched(items: Sequence[Dict[str, Any]], size: int) -> List[Sequence[Dict[str, Any]]]:
    """按 size 切片，size<=0 时不分批。"""
    if size <= 0:
        return [items]
    return [items[i:i + size] for i in range(0, len(items), size)]


def _to_float(value: Any) -> Optional[float]:
    """把分数转 float；NaN/异常一律返回 None（报告里单列 NaN 数）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def evaluate_records(records: Sequence[Dict[str, Any]], cfg: EvalConfig) -> Dict[str, Any]:
    """对记录批跑 RAGAS，返回逐样本分数与总体均值。

    Args:
        records: runner 产出的记录。
        cfg: 评估配置（批量、裁判模型）。

    Returns:
        {
          "per_sample": {qid: {metric: float|None}},
          "overall": {metric: float|None},
          "failures": [错误描述],
          "metric_names": [...],
        }
    """
    _, evaluate = _import_ragas()
    metrics = _build_metrics()
    llm = _build_llm(cfg)
    embeddings = _build_embeddings()

    per_sample: Dict[str, Dict[str, Optional[float]]] = {}
    failures: List[str] = []

    # 1. 分批跑：一次全量塞进去容易 OOM（60 条以上建议每批 20 条）
    for batch_idx, batch in enumerate(_batched(records, cfg.ragas_batch_size), start=1):
        try:
            dataset = _build_dataset(batch)
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                # 单条失败记为 NaN，不要整批炸掉；NaN 在报告里当 bad case
                raise_exceptions=False,
            )
            frame = result.to_pandas()
        except Exception as exc:  # noqa: BLE001 - 单批失败不影响其他批
            failures.append(f"batch {batch_idx}: {type(exc).__name__}: {exc}")
            print(f"[warn] 第 {batch_idx} 批评估失败：{exc}")
            continue

        # 2. 逐行映射回 qid
        metric_cols = [c for c in frame.columns if c in METRIC_NAMES]
        for row_idx, (_, row) in enumerate(frame.iterrows()):
            qid = str(row.get("qid", "")) if "qid" in frame.columns else ""
            if not qid and row_idx < len(batch):
                qid = str(batch[row_idx].get("qid", ""))
            if not qid:
                continue
            per_sample[qid] = {name: _to_float(row.get(name)) for name in metric_cols}

    # 3. 汇总均值（跳过 None，避免"没算"被当成 0）
    overall: Dict[str, Optional[float]] = {}
    for name in METRIC_NAMES:
        values = [v.get(name) for v in per_sample.values()]
        valid = [x for x in values if x is not None]
        overall[name] = (sum(valid) / len(valid)) if valid else None

    return {
        "per_sample": per_sample,
        "overall": overall,
        "failures": failures,
        "metric_names": list(METRIC_NAMES),
    }