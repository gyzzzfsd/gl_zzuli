"""评估报告的生成：CSV 明细、bad case 清单、Markdown 汇总、JSON 快照。

为什么报告要自己写而不是直接 `print(df)`：
1. 评估结果最终要能回答三个问题——"现在多少分 / 比上次好还是坏 / 该改哪一层"；
2. 参数快照与 NaN 统计必须一起出现，否则分数变了也说不清是代码还是配置导致；
3. 输出要能进 git（Markdown + CSV），方便 diff 历史。
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from eval.config import EvalConfig
from eval.metrics_ext import mean_ignoring_none, score_item

# 明细表的固定列序（先元信息，再确定性指标，最后 RAGAS 分数）
BASE_COLUMNS = [
    "qid", "type", "difficulty", "doc", "item_name", "question", "reference",
    "answer", "n_contexts", "n_kg_triples", "used_item_names",
    "citation_count", "elapsed_ms", "error",
]
EXT_METRIC_COLUMNS = [
    "numeric_recall", "numeric_strict", "refusal_accuracy", "over_refusal", "citation_valid_rate",
]

# "越低越好"的指标：不能套用"< 阈值即坏"的规则（over_refusal=0 恰恰是最好的）
LOWER_IS_BETTER = {"over_refusal"}
# 过度拒答的上限：可答题里整段拒答的比例超过它就值得看一眼
OVER_REFUSAL_LIMIT = 0.3

# 指标不达标时的"该改哪一层"提示；键必须与指标名一致
_METRIC_HINT = {
    "faithfulness": "答案在扩写/编造：改 ANSWER_PROMPT（不补原因、只用参考内容里的事实），并检查上下文是否过长导致模型自由发挥",
    "answer_relevancy": "答案跑题或啰嗦：收紧 prompt（2–5 句、覆盖问题点名的对象、不重复问题）",
    "context_precision": "检索噪声大/排序差：调 rerank 截断（top_k / gap 阈值）、检查 RRF 权重与内容去重",
    "context_recall": "该召回的没召回：查切片粒度（max_content_length）、检索 limit、item_name 是否把正确文档挡掉",
    "answer_correctness": "整体与参考答案偏差大：先看是召回问题还是生成问题（对比 context_recall 与 faithfulness）",
    "numeric_recall": "数值/单位缺失或被改写：prompt 允许直接引用数值，必要时加数值强校验（台账）",
    "refusal_accuracy": "该拒答却在编：统一拒答话术，并在 prompt 里要求“语料无依据时必须拒答”",
    "citation_valid_rate": "引用编号越界/编造：校验【来源】里的 [n] 是否落在本次上下文编号范围内，越界就删",
}


def _fmt(value: Optional[float]) -> str:
    """数值格式化：None 显示为 —，其余保留 4 位小数。"""
    return "—" if value is None else f"{value:.4f}"


def enrich(records: Sequence[Dict[str, Any]], items_by_qid: Dict[str, Any],
           ragas_per_sample: Optional[Dict[str, Dict[str, Optional[float]]]] = None) -> List[Dict[str, Any]]:
    """把记录补上指标，形成报告行。

    Args:
        records: runner 产出的记录。
        items_by_qid: {qid: QAItem}，用于取 must_have / type 语义。
        ragas_per_sample: {qid: {metric: 分数}}，可选。

    Returns:
        报告行列表（保持记录顺序）。
    """
    ragas_per_sample = ragas_per_sample or {}
    rows: List[Dict[str, Any]] = []

    for record in records:
        qid = str(record.get("qid", ""))
        row: Dict[str, Any] = {
            "qid": qid,
            "type": record.get("type", ""),
            "difficulty": record.get("difficulty", ""),
            "doc": record.get("doc", ""),
            "item_name": record.get("item_name", ""),
            "question": record.get("question", ""),
            "reference": record.get("reference", ""),
            "answer": record.get("answer", ""),
            "n_contexts": record.get("n_contexts", 0),
            "n_kg_triples": record.get("n_kg_triples", 0),
            "used_item_names": ";".join(record.get("item_names_used") or []),
            "elapsed_ms": record.get("elapsed_ms", 0),
            "error": record.get("error", ""),
        }

        # 1. 确定性指标：题目不在 items 里（CSV 改过）就跳过，避免 KeyError
        item = items_by_qid.get(qid)
        ext = score_item(item, record) if item else {}
        for name in EXT_METRIC_COLUMNS:
            row[name] = ext.get(name)
        row["citation_count"] = ext.get("citation_count", 0)

        # 2. RAGAS 分数（可能没有）
        for name, value in (ragas_per_sample.get(qid) or {}).items():
            row[name] = value

        rows.append(row)

    return rows


def _ragas_metric_names() -> List[str]:
    """RAGAS 五项指标名（延迟导入，避免 report 模块强依赖 ragas_adapter 的路径逻辑）。"""
    from eval.ragas_adapter import METRIC_NAMES

    return list(METRIC_NAMES)


def _metric_names(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """汇总所有出现过的指标列（保持固定顺序，未知的排最后）。"""
    present = {key for row in rows for key in row}
    ordered = [name for name in EXT_METRIC_COLUMNS if name in present]
    ordered += [name for name in _ragas_metric_names() if name in present]
    return ordered


def _summarize_group(rows: Sequence[Dict[str, Any]], names: Sequence[str]) -> Dict[str, Optional[float]]:
    """对一组行求各指标均值。"""
    return {name: mean_ignoring_none([row.get(name) for row in rows]) for name in names}


def summarize(rows: Sequence[Dict[str, Any]], cfg: EvalConfig, mode: str,
              config_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """生成 summary 字典（JSON 快照 + 报告依据）。

    Args:
        rows: enrich 后的行。
        cfg: 评估配置（含门槛）。
        mode: 运行模式，写进快照便于区分。
        config_snapshot: 关键参数快照。

    Returns:
        summary 字典。
    """
    names = _metric_names(rows)

    # 1. 总体与分层
    overall = _summarize_group(rows, names)
    by_type: Dict[str, Dict[str, Optional[float]]] = {}
    types = sorted({str(row.get("type", "")) for row in rows})
    for qtype in types:
        group = [row for row in rows if str(row.get("type", "")) == qtype]
        by_type[qtype] = _summarize_group(group, names)

    # 2. 质量诊断计数
    nan_counts = {
        name: sum(1 for row in rows if row.get(name) is None)
        for name in names
    }
    error_rows = [row for row in rows if row.get("error")]
    # 空上下文分两种：可答题空 = 故障；不可答题空 = 正常（本来就没证据）
    empty_context = [row for row in rows
                     if not row.get("n_contexts") and row.get("type") != "unanswerable"]
    empty_expected = [row for row in rows
                      if not row.get("n_contexts") and row.get("type") == "unanswerable"]

    # 3. 门槛判定
    below = [name for name in names
             if overall.get(name) is not None and overall[name] < cfg.thresholds.get(name, -1)]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "csv": cfg.csv_path.name,
        "mode": mode,
        "sample_count": len(rows),
        "thresholds": dict(cfg.thresholds),
        "overall": overall,
        "by_type": by_type,
        "nan_counts": nan_counts,
        "error_count": len(error_rows),
        "empty_context_count": len(empty_context),
        "empty_context_unanswerable_count": len(empty_expected),
        "below_threshold": below,
        "config_snapshot": config_snapshot or {},
        "elapsed_total_ms": sum(int(row.get("elapsed_ms") or 0) for row in rows),
    }


def _worst_type(summary: Dict[str, Any], metric: str) -> Optional[str]:
    """找该指标最差的题型（用于给出更具体的建议）。"""
    candidates = [
        (qtype, scores.get(metric))
        for qtype, scores in (summary.get("by_type") or {}).items()
        if scores.get(metric) is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda pair: pair[1])[0]


def _bad_sort_key(row: Dict[str, Any], names: Sequence[str]) -> float:
    """bad case 排序键：可用指标均值越小越靠前；有错误/空上下文的排最前。"""
    penalty = 0.0
    if row.get("error"):
        penalty -= 1.0
    if not row.get("n_contexts"):
        penalty -= 0.5
    valid = [row.get(name) for name in names
             if row.get(name) is not None and name not in LOWER_IS_BETTER]
    score = (sum(valid) / len(valid)) if valid else 0.0
    return penalty + score


def write_all(rows: Sequence[Dict[str, Any]], summary: Dict[str, Any],
              cfg: EvalConfig, baseline: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """写出全部报告文件。

    Args:
        rows: enrich 后的行。
        summary: summarize 的结果。
        cfg: 评估配置。
        baseline: 基线 summary（可选），用于算差值。

    Returns:
        {产物名: 路径}。
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    names = _metric_names(rows)

    # 1. results.csv（Excel 直接打开不乱码，用 utf-8-sig）
    results_path = cfg.out_dir / "results.csv"
    columns = BASE_COLUMNS + names
    with results_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: ("" if row.get(col) is None else row.get(col)) for col in columns})

    # 2. bad_cases.csv
    bad_threshold = cfg.bad_score_threshold
    # 只有"任何样本都该算出来"的指标（RAGAS 五项）才把 NaN 当故障；
    # 数值/拒答/引用这类指标天然按题型不适用（无数值题、可答题），NaN 是正常的
    always_metrics = set(_ragas_metric_names())
    bad_rows = []
    for row in rows:
        too_low = any(row.get(name) is not None and row[name] < bad_threshold
                      for name in names if name not in LOWER_IS_BETTER)
        too_high = any(row.get(name) is not None and row[name] > OVER_REFUSAL_LIMIT
                       for name in names if name in LOWER_IS_BETTER)
        has_nan = any(row.get(name) is None for name in names if name in always_metrics)
        if row.get("error") or too_low or too_high or has_nan:
            bad_rows.append(row)
    bad_rows.sort(key=lambda row: _bad_sort_key(row, names))

    bad_path = cfg.out_dir / "bad_cases.csv"
    with bad_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in bad_rows:
            writer.writerow({col: ("" if row.get(col) is None else row.get(col)) for col in columns})

    # 3. summary.json
    summary_path = cfg.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4. summary.md
    md_path = cfg.out_dir / "summary.md"
    md_path.write_text(_render_markdown(rows, summary, bad_rows, baseline, names), encoding="utf-8")

    return {"results": results_path, "bad_cases": bad_path,
            "summary_json": summary_path, "summary_md": md_path}


def _render_markdown(rows: Sequence[Dict[str, Any]], summary: Dict[str, Any],
                     bad_rows: Sequence[Dict[str, Any]], baseline: Optional[Dict[str, Any]],
                     names: Sequence[str]) -> str:
    """渲染 Markdown 报告。

    Args:
        rows: 全部行。
        summary: 汇总。
        bad_rows: bad case（已排序）。
        baseline: 基线 summary 或 None。
        names: 指标名列表。

    Returns:
        Markdown 文本。
    """
    lines: List[str] = []
    overall = summary["overall"]
    base_overall = (baseline or {}).get("overall") or {}

    lines.append(f"# 评估报告 {summary['generated_at']}")
    lines.append("")
    lines.append(f"- 数据集：`{summary['csv']}`（{summary['sample_count']} 条）")
    lines.append(f"- 模式：`{summary['mode']}`")
    lines.append(f"- 总耗时：{summary['elapsed_total_ms'] / 1000:.1f}s")
    lines.append(f"- 错误样本：{summary['error_count']} 条｜上下文为空：{summary['empty_context_count']} 条"
                 f"（另有不可答题 {summary.get('empty_context_unanswerable_count', 0)} 条，属正常）")
    lines.append("")

    # 1. 总体指标
    lines.append("## 1 总体指标")
    lines.append("")
    lines.append("| 指标 | 本次 | 基线 | 差值 | 门槛 | 结论 |")
    lines.append("|---|---|---|---|---|---|")
    for name in names:
        value = overall.get(name)
        base = base_overall.get(name)
        delta = (value - base) if (value is not None and base is not None) else None
        threshold = summary["thresholds"].get(name)
        if value is None:
            verdict = "未计算"
        elif threshold is None:
            verdict = "（无门槛）"
        elif value < threshold:
            verdict = "**不达标**"
        else:
            verdict = "达标"
        lines.append(
            f"| `{name}` | {_fmt(value)} | {_fmt(base)} | "
            f"{'—' if delta is None else f'{delta:+.4f}'} | "
            f"{'—' if threshold is None else f'{threshold:.2f}'} | {verdict} |"
        )
    lines.append("")

    # 2. 分层
    lines.append("## 2 分层指标（按题型）")
    lines.append("")
    header = "| 题型 | 条数 | " + " | ".join(f"`{n}`" for n in names) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(names) + 2))
    for qtype, scores in (summary.get("by_type") or {}).items():
        count = sum(1 for row in rows if str(row.get("type", "")) == qtype)
        lines.append(f"| {qtype} | {count} | " + " | ".join(_fmt(scores.get(n)) for n in names) + " |")
    lines.append("")

    # 3. NaN 统计
    lines.append("## 3 未计算（NaN）统计")
    lines.append("")
    nan_counts = summary["nan_counts"]
    if all(count == 0 for count in nan_counts.values()):
        lines.append("全部指标均算出，无 NaN。")
    else:
        lines.append("| 指标 | NaN 条数 |")
        lines.append("|---|---|")
        for name, count in nan_counts.items():
            if count:
                lines.append(f"| `{name}` | {count} |")
        lines.append("")
        lines.append("> NaN 通常意味着上下文为空、裁判调用失败或该题不适用（如无数值题）。"
                     "空上下文要当硬故障处理，不能只当低分。")
    lines.append("")

    # 4. bad case
    lines.append(f"## 4 Bad case（{len(bad_rows)} 条，按分数升序）")
    lines.append("")
    if not bad_rows:
        lines.append("无。")
    else:
        lines.append("| qid | 题型 | 问题 | 上下文数 | 最低指标 | 错误 |")
        lines.append("|---|---|---|---|---|---|")
        for row in bad_rows[:15]:
            valid = [(name, row.get(name)) for name in names
                     if row.get(name) is not None and name not in LOWER_IS_BETTER]
            worst = min(valid, key=lambda pair: pair[1]) if valid else ("—", None)
            question = str(row.get("question", ""))[:28]
            error = str(row.get("error", ""))[:32]
            lines.append(
                f"| {row.get('qid')} | {row.get('type')} | {question} | {row.get('n_contexts')} | "
                f"`{worst[0]}`={_fmt(worst[1])} | {error} |"
            )
    lines.append("")

    # 5. 结论与建议
    lines.append("## 5 结论与建议")
    lines.append("")
    below = summary.get("below_threshold") or []
    if not below:
        lines.append("- 所有指标达到门槛。")
    else:
        for name in below:
            worst_type = _worst_type(summary, name)
            hint = _METRIC_HINT.get(name, "（暂无建议模板）")
            suffix = f"；最差题型：`{worst_type}`" if worst_type else ""
            lines.append(f"- **`{name}` = {_fmt(overall.get(name))}** 低于门槛：{hint}{suffix}")
    if summary["error_count"]:
        lines.append(f"- 有 {summary['error_count']} 条样本执行报错，先修链路再看分数（见 bad_cases.csv 的 error 列）。")
    if summary["empty_context_count"]:
        lines.append(f"- 有 {summary['empty_context_count']} 条样本上下文为空：优先怀疑 item_name 过滤过严或该主体未入库。")
    if base_overall:
        regressions = [name for name in names
                       if overall.get(name) is not None and base_overall.get(name) is not None
                       and overall[name] < base_overall[name] - 0.02]
        if regressions:
            lines.append(f"- 相对基线回归：{', '.join(f'`{name}`' for name in regressions)}")
    lines.append("")

    # 6. 参数快照
    snapshot = summary.get("config_snapshot") or {}
    if snapshot:
        lines.append("## 6 参数快照（本次运行实际生效）")
        lines.append("")
        lines.append("| 参数 | 值 |")
        lines.append("|---|---|")
        for key, value in snapshot.items():
            lines.append(f"| `{key}` | {value} |")
        lines.append("")

    return "\n".join(lines) + "\n"