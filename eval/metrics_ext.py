"""确定性指标：数值命中、拒答正确、引用有效。

为什么要有这一层：RAGAS 的 answer_correctness 是语义相似度，对"94.6% 写成 96.4%"
几乎不敏感；而生物医学与设备场景里，数值错一位就是错答案。这三个指标不调 LLM，
零成本、可复现、能进 CI。

三个指标的判定口径：
1. numeric_recall：参考答案里该出现的数值，答案里出现了多少（按数值集合算，支持部分命中）；
2. numeric_strict：该出现的数值全部出现才算 1（更严，用于验收）；
3. refusal_accuracy：只在 unanswerable 样本上算——该拒答时是否拒答了；
4. citation_valid_rate：`【来源】` 区块里的 [n] 是否都落在本次上下文编号范围内。
   没有引用区块时返回 None（不参与平均），避免"没做引用"被算成"引用全对"。
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Sequence

from eval.dataset import QAItem

# 答案末尾的附加区块：评估前要截掉，否则会污染语义类指标
ATTACHMENT_MARKERS = ("【图片】", "【来源】")

# 拒答话术：与 prompt 里约定的"现有资料未提及……"保持一致，命中任一即视为拒答
# 覆盖三类常见措辞：直接说没有（未提及/未涵盖）、说资料不够（资料不足/信息不足）、
# 说无法判断（无法回答/无法确定），避免把"礼貌拒答"漏判成"胡编"
REFUSAL_PATTERN = re.compile(
    r"(未提及|未涵盖|未说明|未给出|未涉及|未提供"
    r"|资料不足|信息不足|数据不足"
    r"|暂无(?:相关)?(?:资料|信息|数据)"
    r"|没有(?:相关)?(?:资料|内容|说明|记录)"
    r"|无法从(?:参考|现有)(?:内容|资料)"
    r"|无法(?:回答|确定|判断)"
    r"|资料(?:中|里)?(?:并)?未)"
)

# 数值：可带负号与小数；百分比、单位不纳入 token（单位错在 answer_correctness 里体现）
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

# 【来源】区块里的引用编号：只在这个区块内解析，避免把"第 3 步"误判成引用
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def strip_attachments(answer: str) -> str:
    """截掉答案末尾的【图片】/【来源】区块，只留正文。

    Args:
        answer: 模型原始输出。

    Returns:
        正文部分（去掉尾随空白）。
    """
    head = answer or ""
    for marker in ATTACHMENT_MARKERS:
        head = head.split(marker, 1)[0]
    return head.strip()


def extract_numbers(text: str) -> List[str]:
    """抽取文本中的数值并归一化（去千分位、去尾随零、全角转半角）。

    为什么归一化：`0.90` 与 `0.9`、`１００` 与 `100` 是同一个数，
    不归一会让"数值没命中"变成误报。

    Args:
        text: 任意文本。

    Returns:
        归一化后的数值字符串列表（保序、含重复，便于做多重集比较）。
    """
    normalized = unicodedata.normalize("NFKC", text or "")
    # 千分位要先合并再分词，否则 "1,000" 会被拆成 "1" 和 "000"
    while True:
        merged = re.sub(r"(?<=\d),(?=\d)", "", normalized)
        if merged == normalized:
            break
        normalized = merged
    numbers: List[str] = []
    for raw in NUMBER_PATTERN.findall(normalized):
        token = raw.replace(",", "")
        try:
            value = Decimal(token).normalize()
        except InvalidOperation:
            numbers.append(token)
            continue
        rendered = format(value, "f")
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        numbers.append(rendered)
    return numbers


def reference_numbers(item: QAItem) -> List[str]:
    """取"该出现的数值"。

    优先用 must_have 里的数值 token（人工指定，干净）；
    没有时退回参考答案全文抽数（会有 M1、第 3 步这类噪声，所以建集时建议填 must_have）。

    Args:
        item: 一条样本。

    Returns:
        归一化后的数值列表。
    """
    from_must = [num for token in item.must_have for num in extract_numbers(token)]
    if from_must:
        return from_must
    return extract_numbers(item.reference)


def numeric_scores(item: QAItem, answer_body: str) -> Dict[str, Optional[float]]:
    """计算数值命中。

    Args:
        item: 样本（提供 must_have / 参考答案）。
        answer_body: 已剥离附加区块的答案正文。

    Returns:
        {"numeric_recall": 0~1 或 None, "numeric_strict": 0/1 或 None}；
        参考答案里没有数值时两者均为 None（不参与平均）。
    """
    expected = reference_numbers(item)
    if not expected:
        return {"numeric_recall": None, "numeric_strict": None}

    # 1. 答案里的数值做成多重集，避免同一个数被算两次命中
    pool: Dict[str, int] = {}
    for num in extract_numbers(answer_body):
        pool[num] = pool.get(num, 0) + 1

    # 2. 逐个消耗
    hit = 0
    for num in expected:
        if pool.get(num, 0) > 0:
            hit += 1
            pool[num] -= 1

    recall = hit / len(expected)
    return {"numeric_recall": recall, "numeric_strict": 1.0 if hit == len(expected) else 0.0}


def is_refusal(answer_body: str) -> bool:
    """判断答案是否属于拒答。

    Args:
        answer_body: 已剥离附加区块的答案正文。

    Returns:
        True 表示命中拒答话术。
    """
    return bool(REFUSAL_PATTERN.search(answer_body or ""))


def refusal_scores(item: QAItem, answer_body: str) -> Dict[str, Optional[float]]:
    """计算拒答正确性。

    只在 unanswerable 样本上给分；其余样本返回 None，但额外给出 over_refusal 诊断位。

    Args:
        item: 样本。
        answer_body: 答案正文。

    Returns:
        {"refusal_accuracy": 0/1 或 None, "over_refusal": 0/1}
    """
    refused = is_refusal(answer_body)
    if item.type == "unanswerable":
        # 不可答题只关心"该拒答时拒答了"，over_refusal 不适用（返回 None 不参与平均）
        return {"refusal_accuracy": 1.0 if refused else 0.0, "over_refusal": None}

    # 可答题却整段拒答：单独记一个诊断指标，不进主门槛（部分拒答是合理的）
    over = 1.0 if (refused and not extract_numbers(answer_body)) else 0.0
    return {"refusal_accuracy": None, "over_refusal": over}


def parse_citation_ids(answer: str) -> List[int]:
    """解析【来源】区块里的引用编号。

    Args:
        answer: 模型原始输出（含附加区块）。

    Returns:
        去重保序的编号列表；没有【来源】区块时返回空列表。
    """
    # 1. 只取【来源】之后的部分
    if "【来源】" not in (answer or ""):
        return []
    block = answer.split("【来源】", 1)[1]

    # 2. 遇到下一个区块标记就停
    for marker in ("【图片】",):
        block = block.split(marker, 1)[0]

    ids: List[int] = []
    for raw in CITATION_PATTERN.findall(block):
        value = int(raw)
        if value not in ids:
            ids.append(value)
    return ids


def citation_score(answer: str, n_contexts: int) -> Dict[str, Optional[float]]:
    """计算引用有效率。

    Args:
        answer: 模型原始输出。
        n_contexts: 本次放进 prompt 的上下文条数（编号上界）。

    Returns:
        {"citation_count": 引用条数, "citation_valid_rate": 0~1 或 None}。
    """
    ids = parse_citation_ids(answer)
    if not ids:
        return {"citation_count": 0, "citation_valid_rate": None}

    # 编号必须落在 [1, n_contexts]，越界说明模型编了引用
    valid = sum(1 for i in ids if 1 <= i <= n_contexts)
    return {"citation_count": len(ids), "citation_valid_rate": valid / len(ids)}


def score_item(item: QAItem, record: Dict) -> Dict[str, Optional[float]]:
    """汇总一条样本的全部确定性指标。

    Args:
        item: 样本。
        record: runner 产出的记录（含 answer / raw_answer / n_contexts）。

    Returns:
        指标字典，None 表示该样本不适用（不参与平均）。
    """
    body = record.get("answer") or ""
    raw = record.get("raw_answer") or body
    n_contexts = int(record.get("n_contexts") or 0)

    scores: Dict[str, Optional[float]] = {}
    scores.update(numeric_scores(item, body))
    scores.update(refusal_scores(item, body))
    scores.update(citation_score(raw, n_contexts))
    return scores


def mean_ignoring_none(values: Sequence[Optional[float]]) -> Optional[float]:
    """求均值，跳过 None。

    Args:
        values: 指标值序列，允许 None。

    Returns:
        均值；全部为 None 时返回 None（而不是 0，避免"没算"被当成"算出来是 0"）。
    """
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)