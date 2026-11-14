"""无外部依赖自检：不连 Milvus/Neo4j/LLM，验证指标与报告逻辑是否正确。

为什么需要：评估脚本本身出错时，分数会"看起来正常"（全 0 或全 NaN），
比跑不起来更危险。这个文件用断言把易错点钉死，可在 CI 里几秒跑完。

用法：
    python eval/selfcheck.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import report as report_mod  # noqa: E402
from eval.config import EvalConfig  # noqa: E402
from eval.dataset import QAItem, audit_qa, load_qa, stratified_counts  # noqa: E402
from eval.gate import check, save_baseline  # noqa: E402
from eval.metrics_ext import (  # noqa: E402
    extract_numbers,
    is_refusal,
    mean_ignoring_none,
    numeric_scores,
    parse_citation_ids,
    refusal_scores,
    strip_attachments,
)

SAMPLE_CSV = REPO_ROOT / "eval" / "eval_doc" / "qa_sample.csv"


def _item(qid: str = "001-deadbeef", qtype: str = "paper_number") -> QAItem:
    """构造一条测试样本。"""
    return QAItem(
        qid=qid,
        question="外部验证集敏感度和假阳性是多少？",
        reference="敏感度 94.6%，每例假阳性数 1.9。",
        item_name="测试主体",
        doc="P01.md",
        type=qtype,
        must_have=("94.6%", "1.9"),
        difficulty="easy",
    )


def check_numbers() -> None:
    """数值抽取与命中。"""
    # 1. 归一化：全角、尾随零、千分位、负号
    assert extract_numbers("０.９０") == ["0.9"], extract_numbers("０.９０")
    assert extract_numbers("1,000") == ["1000"], extract_numbers("1,000")
    assert extract_numbers("范围 0 至 299") == ["0", "299"]
    assert extract_numbers("-3.50") == ["-3.5"]

    # 2. 全部命中
    scores = numeric_scores(_item(), "敏感度 94.6%，每例假阳性 1.9 个")
    assert scores["numeric_recall"] == 1.0, scores
    assert scores["numeric_strict"] == 1.0, scores

    # 3. 部分命中：recall 给部分分，strict 归零
    partial = numeric_scores(_item(), "敏感度 94.6%")
    assert abs(partial["numeric_recall"] - 0.5) < 1e-9, partial
    assert partial["numeric_strict"] == 0.0, partial

    # 4. 数值错位：写成 96.4% 必须判 0
    wrong = numeric_scores(_item(), "敏感度 96.4%，每例假阳性 1.9 个")
    assert wrong["numeric_strict"] == 0.0, wrong

    # 5. 参考答案里没有数值时不参与平均
    no_number = QAItem("x", "q", "见说明书", "i", "d", "manual_step", ())
    assert numeric_scores(no_number, "见说明书")["numeric_recall"] is None


def check_refusal() -> None:
    """拒答识别与计分。"""
    assert is_refusal("现有资料未提及该功能")
    assert is_refusal("参考内容未涵盖此参数")
    assert not is_refusal("血压测量范围是 0 至 299 毫米汞柱")

    unanswerable = QAItem("y", "q", "现有资料未提及", "i", "d", "unanswerable", ())
    assert refusal_scores(unanswerable, "现有资料未提及该功能")["refusal_accuracy"] == 1.0
    assert refusal_scores(unanswerable, "支持蓝牙")["refusal_accuracy"] == 0.0

    answerable = _item()
    scores = refusal_scores(answerable, "现有资料未提及")
    assert scores["refusal_accuracy"] is None
    assert scores["over_refusal"] == 1.0


def check_attachments_and_citation() -> None:
    """附加区块剥离与引用校验。"""
    assert strip_attachments("正文内容\n【图片】\n<http://x/1.png>") == "正文内容"
    assert strip_attachments("正文内容【来源】\n[1] M01") == "正文内容"

    assert parse_citation_ids("无引用") == []
    ids = parse_citation_ids("答案\n【来源】\n[1] M01 > 规格\n[3] P01 > 结果")
    assert ids == [1, 3], ids

    # 只给 2 条上下文，第 3 条属于编造 → 有效率 0.5
    from eval.metrics_ext import citation_score

    score = citation_score("答案\n【来源】\n[1] a\n[3] b", n_contexts=2)
    assert score["citation_count"] == 2
    assert abs(score["citation_valid_rate"] - 0.5) < 1e-9, score


def check_dataset() -> None:
    """数据集加载与稳定性。"""
    items = load_qa(SAMPLE_CSV)
    assert len(items) == 6, len(items)
    assert sum(stratified_counts(items).values()) == 6
    assert isinstance(audit_qa(items), list)

    # qid 必须跨进程稳定：两次加载结果一致
    again = load_qa(SAMPLE_CSV)
    assert [i.qid for i in items] == [i.qid for i in again]

    # limit 生效
    assert len(load_qa(SAMPLE_CSV, limit=2)) == 2

    # 缺列必须报错，而不是静默当成空
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.csv"
        bad.write_text("question,answer\nq,a\n", encoding="utf-8")
        try:
            load_qa(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("缺列时应抛 ValueError")


def check_report_and_gate() -> None:
    """报告生成与门禁判定。"""
    items = load_qa(SAMPLE_CSV)
    records = []
    for idx, item in enumerate(items):
        # 前两条故意做成"好答案"，其余做成"坏答案"，便于断言 bad case 有内容
        good = idx < 2
        records.append({
            "qid": item.qid,
            "question": item.question,
            "reference": item.reference,
            "item_name": item.item_name,
            "doc": item.doc,
            "type": item.type,
            "difficulty": item.difficulty,
            "answer": item.reference if good else "资料不足",
            "raw_answer": (item.reference if good else "资料不足") + "\n【来源】\n[9] 越界引用",
            "contexts": [item.reference] if good else [],
            "n_contexts": 1 if good else 0,
            "n_kg_triples": 0,
            "item_names_used": [item.item_name],
            "elapsed_ms": 12,
            "error": "" if good else "上下文为空（检索未命中或主体过滤过严）",
        })

    with tempfile.TemporaryDirectory() as tmp:
        cfg = EvalConfig(csv_path=SAMPLE_CSV, out_dir=Path(tmp) / "report", mode="ext")
        rows = report_mod.enrich(records, {item.qid: item for item in items})
        summary = report_mod.summarize(rows, cfg, "ext", {"import.max_content_length": 2000})
        paths = report_mod.write_all(rows, summary, cfg)

        # 1. 四类产物都要落盘
        for key in ("results", "bad_cases", "summary_json", "summary_md"):
            assert paths[key].is_file(), key

        # 2. summary 结构
        assert summary["sample_count"] == 6
        # 空上下文只统计"可答题"：不可答题（第 6 条）空上下文属正常，单列计数
        assert summary["empty_context_count"] == 3, summary["empty_context_count"]
        assert summary["empty_context_unanswerable_count"] == 1
        assert summary["error_count"] == 4
        assert "numeric_recall" in summary["overall"]
        assert summary["config_snapshot"]["import.max_content_length"] == 2000

        # 3. Markdown 有内容、含关键小节
        text = paths["summary_md"].read_text(encoding="utf-8")
        assert "评估报告" in text and "总体指标" in text and "参数快照" in text

        # 4. 门禁：坏结果必须被拦
        failures = check(summary, None, cfg.thresholds, cfg.regression_tolerance)
        assert failures, "明显不达标时应返回失败原因"

        # 5. 基线固化 + 相对回归判定：把基线里某项抬高，当前值应被判定为"下降"
        baseline_path = Path(tmp) / "baseline.json"
        save_baseline(cfg.out_dir, baseline_path)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        assert baseline["sample_count"] == 6

        baseline["overall"]["numeric_recall"] = 1.0
        relaxed = {name: 0.0 for name in summary["overall"]}  # 关掉绝对门槛，只验相对回归
        fails = check(summary, baseline, relaxed, cfg.regression_tolerance)
        assert any("numeric_recall" in f and "下降" in f for f in fails), fails


def check_state_contract() -> None:
    """runner 构造的 state 必须覆盖图上所有"下标访问"的字段。

    为什么单独查一遍：`answer_output` 用 `state["task_id"]` / `state["item_names"]`，
    `item_name_confirm` 用 `state["session_id"]`，缺一个就是 KeyError，
    而 KeyError 会在跑到第 N 条时才出现，代价很高。
    """
    from eval.runner import build_state

    state = build_state(_item())
    required = ("original_query", "session_id", "task_id", "is_stream", "history",
                "item_names", "item_name_locked", "save_history")
    missing = [key for key in required if key not in state]
    assert not missing, f"state 缺字段：{missing}"

    # 评估必须锁定主体、关闭历史写入，否则会短路或污染生产会话
    assert state["item_name_locked"] is True
    assert state["save_history"] is False
    assert state["is_stream"] is False
    assert state["item_names"] == ["测试主体"]

    # 逐条唯一：避免会话历史/任务态互相串味
    assert build_state(_item())["session_id"] != state["session_id"]


def check_mean() -> None:
    """均值跳过 None，而不是当 0。"""
    assert mean_ignoring_none([1.0, None, 0.0]) == 0.5
    assert mean_ignoring_none([None, None]) is None


def main() -> int:
    """依次跑全部自检。

    Returns:
        0 表示全部通过。
    """
    checks = [
        ("数值指标", check_numbers),
        ("拒答指标", check_refusal),
        ("附加区块与引用", check_attachments_and_citation),
        ("数据集", check_dataset),
        ("报告与门禁", check_report_and_gate),
        ("state 契约", check_state_contract),
        ("均值", check_mean),
    ]
    for name, func in checks:
        func()
        print(f"  ✓ {name}")
    print(f"selfcheck 全部通过（{len(checks)} 组）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())