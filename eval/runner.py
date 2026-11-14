"""跑生产查询图，采集每条样本的答案与上下文。

关键约束（决定了这个模块为什么这么写）：
1. 必须走真实的 `query_app`，否则评的不是线上链路；
2. `answer_output` 直接下标取 `task_id`/`session_id`/`item_names`，三个字段缺一不可；
3. 评估不能写 Mongo 历史，也不能被历史影响（否则第 5 题会被第 4 题带偏）；
4. 单条失败不能打断整轮，失败的样本照样落盘，报告里能看见原因；
5. 每跑完一条就把结果 append 到 JSONL，中断后可 `--resume` 续跑，不重复花钱。
"""

from __future__ import annotations

import json
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List

from eval._bootstrap import ensure_backend_on_path
from eval.dataset import QAItem
from eval.metrics_ext import strip_attachments


class BackendUnavailable(RuntimeError):
    """业务代码依赖未安装时抛出，给出"该用哪个解释器"的提示。"""


def _load_query_app():
    """延迟导入查询图。

    Returns:
        LangGraph 编译后的 query_app。

    Raises:
        BackendUnavailable: 缺依赖（没装 backend/requirements.txt）或导入链报错。
    """
    ensure_backend_on_path()
    try:
        from app.processor.query_process.main_graph import query_app
    except ImportError as exc:
        raise BackendUnavailable(
            "导入查询图失败，请用装好 backend/requirements.txt 的解释器运行评估"
            f"（原始错误：{exc}）"
        ) from exc

    return query_app


def build_state(item: QAItem) -> Dict[str, Any]:
    """构造一条样本的初始 state。

    为什么每条都用新的 session/task：隔离会话与任务态，避免上下文串味、
    也避免内存任务表被评估样本撑大。

    Args:
        item: 一条样本。

    Returns:
        可直接喂给 `query_app.invoke` 的 state 字典。
    """
    suffix = uuid.uuid4().hex[:8]
    return {
        "original_query": item.question,
        "session_id": f"eval-{item.qid}-{suffix}",
        "task_id": f"eval-{item.qid}-{suffix}",
        "is_stream": False,
        "history": [],
        # 锁定知识主体：跳过 LLM 识别与向量对齐，保证"检索的是人工指定的那个主体"
        "item_names": [item.item_name],
        "item_name_locked": True,
        # 关闭历史写入：评估样本不进生产会话
        "save_history": False,
    }


def _extract_contexts(result: Dict[str, Any]) -> List[str]:
    """从图结果里取"模型真正看到的证据"。

    取 reranked_docs 而不是 rrf_chunks：前者与拼进 prompt 的是同一份，
    顺序即重排顺序，context_precision 会用到顺序。

    Args:
        result: query_app.invoke 的返回值。

    Returns:
        非空 content 列表。
    """
    contexts: List[str] = []
    for doc in result.get("reranked_docs") or []:
        if not isinstance(doc, dict):
            continue
        content = str(doc.get("content") or "").strip()
        if content:
            contexts.append(content)
    return contexts


def _top_scores(result: Dict[str, Any], limit: int = 5) -> List[float]:
    """取重排后的前 N 个分数，用于事后分析断崖截断是否合理。

    Args:
        result: 图结果。
        limit: 取几个。

    Returns:
        分数列表（跳过 None）。
    """
    scores: List[float] = []
    for doc in (result.get("reranked_docs") or [])[:limit]:
        if isinstance(doc, dict) and doc.get("score") is not None:
            try:
                scores.append(round(float(doc["score"]), 4))
            except (TypeError, ValueError):
                continue
    return scores


def run_one(item: QAItem, query_app) -> Dict[str, Any]:
    """跑一条样本，返回记录（不抛异常）。

    Args:
        item: 一条样本。
        query_app: 已编译的查询图。

    Returns:
        记录字典；失败时 `error` 字段带异常栈摘要，答案为空。
    """
    started = time.perf_counter()
    record: Dict[str, Any] = {
        "qid": item.qid,
        "question": item.question,
        "reference": item.reference,
        "item_name": item.item_name,
        "doc": item.doc,
        "type": item.type,
        "difficulty": item.difficulty,
        "answer": "",
        "raw_answer": "",
        "contexts": [],
        "n_contexts": 0,
        "n_kg_triples": 0,
        "top_scores": [],
        "rewritten_query": "",
        "item_names_used": [],
        "elapsed_ms": 0,
        "error": "",
    }

    try:
        # 1. 跑图（同步模式，答案会落在 state["answer"] 上）
        result = query_app.invoke(build_state(item)) or {}

        # 2. 采集
        raw_answer = str(result.get("answer") or "")
        record["raw_answer"] = raw_answer
        record["answer"] = strip_attachments(raw_answer)
        record["contexts"] = _extract_contexts(result)
        record["n_contexts"] = len(record["contexts"])
        record["n_kg_triples"] = len(result.get("kg_triples") or [])
        record["top_scores"] = _top_scores(result)
        record["rewritten_query"] = str(result.get("rewritten_query") or "")
        record["item_names_used"] = list(result.get("item_names") or [])

        # 3. 运行时可诊断的空上下文：不算异常，但报告要能一眼看到
        #    不可答题本来就不该有证据，空上下文对它是正常的，不能记成故障
        if not record["contexts"] and item.type != "unanswerable":
            record["error"] = "上下文为空（检索未命中或主体过滤过严）"

    except Exception as exc:  # noqa: BLE001 - 评估场景要吞掉单条异常，保住整轮
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=4)
    finally:
        record["elapsed_ms"] = int((time.perf_counter() - started) * 1000)

    return record


def release_task() -> None:
    """清理评估产生的任务态与 SSE 队列。

    项目里 `clear_task` 从未被调用（缺陷 D09），连跑几十条会把内存任务表撑大，
    所以评估每跑完一条就清一次；只清 `eval-` 前缀，不碰生产任务。
    """
    try:
        ensure_backend_on_path()
        from app.utils import task_util

        # 这四个表就是任务态的全部落脚点（见 backend/app/utils/task_util.py:5-11）
        for name in ("_tasks_running_list", "_tasks_done_list", "_tasks_result", "_tasks_status"):
            table = getattr(task_util, name, None)
            if isinstance(table, dict):
                for key in [k for k in list(table.keys()) if str(k).startswith("eval-")]:
                    table.pop(key, None)

        # SSE 队列同理：按 task_id 建 key，不清会一直挂在全局字典里
        from app.utils import sse_util

        stream = getattr(sse_util, "_task_stream", None)
        if isinstance(stream, dict):
            for key in [k for k in list(stream.keys()) if str(k).startswith("eval-")]:
                stream.pop(key, None)
    except Exception:
        # 清理失败不影响评估结果（内存表最终随进程结束释放）
        pass


def run_samples(items: List[QAItem], records_path: Path, resume: bool = False) -> List[Dict[str, Any]]:
    """逐条跑样本，边跑边落盘。

    Args:
        items: 样本列表。
        records_path: JSONL 落盘路径。
        resume: True 时跳过文件里已存在的 qid。

    Returns:
        全部记录（含本次新跑的与文件里已有的）。
    """
    # 1. 读取已有结果（断点续跑）
    records: List[Dict[str, Any]] = []
    done: set = set()
    if resume and records_path.is_file():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(row)
            done.add(row.get("qid"))

    # 2. 准备查询图（只加载一次，避免每条都重建模型）
    query_app = _load_query_app()
    records_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. 逐条执行并立即落盘
    with records_path.open("a", encoding="utf-8") as fp:
        for idx, item in enumerate(items, start=1):
            if item.qid in done:
                print(f"[{idx}/{len(items)}] 跳过（已有结果）：{item.question[:36]}")
                continue

            record = run_one(item, query_app)
            records.append(record)
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            fp.flush()

            status = "OK" if not record["error"] else f"ERR({record['error'][:40]})"
            print(f"[{idx}/{len(items)}] {status} ctx={record['n_contexts']} "
                  f"{record['elapsed_ms']}ms :: {item.question[:36]}")

            release_task()

    return records


def collect_config_snapshot() -> Dict[str, Any]:
    """采集当前生效的关键参数，写进报告。

    为什么需要：同一个数据集两次跑的差异，可能来自参数而不是代码；
    报告里带上参数快照，才能回答"这次为什么变好了"。

    Returns:
        参数字典；取不到时对应值为 "unavailable"。
    """
    snapshot: Dict[str, Any] = {}
    try:
        ensure_backend_on_path()
        from app.processor.import_process.config import get_config as get_import_config
        from app.processor.query_process.config import get_config as get_query_config

        imp = get_import_config()
        query = get_query_config()

        # 1. 切片相关（决定召回粒度）
        for name in ("max_content_length", "min_content_length", "overlap_sentences",
                     "chunks_collection", "item_name_collection", "image_caption_collection"):
            if hasattr(imp, name):
                snapshot[f"import.{name}"] = getattr(imp, name)

        # 2. 检索与重排相关（决定召回量与截断）
        for name in ("rerank_max_top_k", "rerank_min_top_k", "rerank_gap_abs", "rerank_gap_ratio",
                     "rrf_k", "rrf_max_results", "rrf_kg_weight", "embedding_search_limit",
                     "hyde_search_limit", "max_context_chars", "item_name_max_options",
                     "kg_entity_align_min_score"):
            if hasattr(query, name):
                snapshot[f"query.{name}"] = getattr(query, name)

        # 3. 模型相关
        import os
        snapshot["env.MODEL"] = os.getenv("MODEL", "")
        snapshot["env.ITEM_MODEL"] = os.getenv("ITEM_MODEL", "")
        snapshot["env.KG_MODEL"] = os.getenv("KG_MODEL", "")
    except Exception as exc:  # noqa: BLE001 - 快照失败不能影响评估本身
        snapshot["error"] = f"{type(exc).__name__}: {exc}"

    return snapshot