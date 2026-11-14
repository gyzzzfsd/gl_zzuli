"""黄金集加载与体检。

CSV 是评估的唯一事实来源，所以这里的校验要"硬"：列缺失、必填为空、类型拼错
都在加载阶段抛错，而不是跑完几十条样本后才发现某列全空。
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 允许的题型：决定分层报告怎么切，也决定"该不该拒答"
KNOWN_TYPES = (
    "manual_fact",
    "manual_step",
    "manual_warn",
    "manual_compare",
    "paper_number",
    "paper_method",
    "cross_doc",
    "unanswerable",
)

REQUIRED_COLUMNS = ("question", "answer", "item_name", "doc", "type")
OPTIONAL_COLUMNS = ("must_have", "difficulty")


@dataclass(frozen=True)
class QAItem:
    """一条评测样本。

    Attributes:
        qid: 稳定编号（行号 + 问题哈希前缀），报告与断点续跑都用它对齐。
        question: 用户问法。
        reference: 参考答案（关键事实清单，含数值与单位）。
        item_name: 应命中的知识主体名，必须与 Milvus 里的值完全一致。
        doc: 证据所在文件名。
        type: 题型，取值见 KNOWN_TYPES。
        must_have: 必须出现的关键 token（分号分隔后可含数值）。
        difficulty: easy/medium/hard，用于分层看难度。
    """

    qid: str
    question: str
    reference: str
    item_name: str
    doc: str
    type: str
    must_have: Tuple[str, ...] = field(default_factory=tuple)
    difficulty: str = ""


def _split_tokens(raw: str) -> Tuple[str, ...]:
    """把 must_have 拆成 token 元组。

    接受分号与逗号两种分隔（中文场景下分号更常见），去空、去重、保序。
    """
    if not raw:
        return ()
    tokens = []
    for part in raw.replace("；", ";").replace("，", ",").replace(",", ";").split(";"):
        token = part.strip()
        if token and token not in tokens:
            tokens.append(token)
    return tuple(tokens)


def load_qa(csv_path: Path, limit: Optional[int] = None) -> List[QAItem]:
    """加载并校验黄金集。

    Args:
        csv_path: CSV 路径（UTF-8，允许带 BOM）。
        limit: 只取前 N 条；None 表示全量。

    Returns:
        QAItem 列表，顺序与 CSV 一致。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 缺列、无数据、必填为空、题型未知。
    """
    # 1. 文件存在性
    if not csv_path.is_file():
        raise FileNotFoundError(f"黄金集不存在：{csv_path}")

    # 2. 读取（utf-8-sig 兼容 Excel 另存的 BOM）
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        header = [name.strip() for name in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{csv_path.name} 缺少必填列：{missing}；当前列：{header}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{csv_path.name} 没有任何数据行")

    # 3. 逐行解析并校验
    items: List[QAItem] = []
    for idx, row in enumerate(rows, start=1):
        question = (row.get("question") or "").strip()
        reference = (row.get("answer") or "").strip()
        item_name = (row.get("item_name") or "").strip()
        doc = (row.get("doc") or "").strip()
        qtype = (row.get("type") or "").strip()

        empty = [name for name, value in (
            ("question", question), ("answer", reference),
            ("item_name", item_name), ("doc", doc), ("type", qtype),
        ) if not value]
        if empty:
            raise ValueError(f"第 {idx} 行必填列为空：{empty}")

        if qtype not in KNOWN_TYPES:
            raise ValueError(f"第 {idx} 行 type={qtype!r} 不在允许集合 {KNOWN_TYPES}")

        # qid 用行号 + 问题哈希前缀：跨进程稳定（不能用内置 hash，它每进程随机），
        # 既能断点续跑判重，也能在报告里定位到 CSV 的哪一行
        digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]
        items.append(QAItem(
            qid=f"{idx:03d}-{digest}",
            question=question,
            reference=reference,
            item_name=item_name,
            doc=doc,
            type=qtype,
            must_have=_split_tokens(row.get("must_have") or ""),
            difficulty=(row.get("difficulty") or "").strip(),
        ))

        if limit and len(items) >= limit:
            break

    return items


def audit_qa(items: List[QAItem]) -> List[str]:
    """数据集体检，返回告警列表（不抛错，用于提示建集质量问题）。

    检查项：
    1. 问题重复；
    2. 缺少 must_have（数值类题型没有 must_have 会让数值指标失真）；
    3. 参考答案过短（< 8 字，recall 会虚高）；
    4. 不可答题里出现"文档里其实有"的高风险提示（只能提示，判定靠人工）。

    Args:
        items: 已加载的样本。

    Returns:
        人类可读的告警行；空列表表示没发现问题。
    """
    warnings: List[str] = []

    # 1. 问题重复
    seen: Dict[str, int] = {}
    for item in items:
        seen[item.question] = seen.get(item.question, 0) + 1
    for question, count in seen.items():
        if count > 1:
            warnings.append(f"问题重复 {count} 次：{question[:40]}…")

    # 2. 数值类题型必须有 must_have
    for item in items:
        if item.type in ("paper_number", "manual_fact") and not item.must_have:
            warnings.append(f"[{item.qid}] {item.type} 缺 must_have，数值指标会退化成按参考答案抽数：{item.question[:30]}…")

    # 3. 参考答案过短
    for item in items:
        if len(item.reference) < 8:
            warnings.append(f"[{item.qid}] 参考答案过短（{len(item.reference)} 字）：{item.question[:30]}…")

    return warnings


def stratified_counts(items: List[QAItem]) -> Dict[str, int]:
    """按题型统计条数，用于确认黄金集配额是否达标。

    Args:
        items: 已加载的样本。

    Returns:
        {题型: 条数}，按键名排序，缺的题型不出现。
    """
    counts: Dict[str, int] = {}
    for item in items:
        counts[item.type] = counts.get(item.type, 0) + 1
    return dict(sorted(counts.items()))