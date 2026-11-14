"""评估配置：路径、阈值、批量、裁判模型。

设计原则：
1) 阈值只写一处，报告与门禁共用，避免"报告说达标、门禁说没达标"；
2) 数值型参数集中在 dataclass 默认值里，需要临时改时用命令行参数覆盖，不散落在脚本里；
3) 只读少数几个有意义的环境变量（裁判模型），其余显式默认，缺关键项不静默失败。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

# 指标门槛：低于它即为不达标；报告里逐项标红，门禁据此拦截
# 说明：前 5 项是 RAGAS 五项；后 3 项是确定性指标（不调 API，可进 CI）
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.70,
    "context_precision": 0.80,
    "context_recall": 0.85,
    "answer_correctness": 0.70,
    "numeric_recall": 0.90,
    "refusal_accuracy": 0.90,
    "citation_valid_rate": 0.95,
}

# 相对基线的允许跌幅：超过即判定回归
REGRESSION_TOLERANCE = 0.02


@dataclass
class EvalConfig:
    """一次评估的全部参数。

    Attributes:
        csv_path: 黄金集路径。
        out_dir: 报告输出目录，本次评估的所有产物都落在这里。
        limit: 只跑前 N 条（冒烟用）；None 表示全量。
        mode: ext（只跑确定性指标）/ ragas（只跑 RAGAS）/ all（两者都跑）。
        ragas_batch_size: RAGAS 分批大小，防止整批塞进内存。
        judge_model: 裁判模型名；空字符串表示沿用生成模型（不建议，自评会偏高）。
        bad_score_threshold: bad case 的分数阈值，低于它进 bad_cases.csv。
        thresholds: 指标门槛，默认取 DEFAULT_THRESHOLDS。
        regression_tolerance: 相对基线的允许跌幅。
    """

    csv_path: Path
    out_dir: Path
    limit: Optional[int] = None
    mode: str = "ext"
    ragas_batch_size: int = 20
    judge_model: str = ""
    bad_score_threshold: float = 0.5
    thresholds: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    regression_tolerance: float = REGRESSION_TOLERANCE

    @classmethod
    def from_args(cls, csv_path: str, out_dir: str, **kwargs) -> "EvalConfig":
        """从命令行参数构造配置。

        Args:
            csv_path: 黄金集路径（相对仓库根目录或绝对路径）。
            out_dir: 输出目录（相对仓库根目录或绝对路径）。
            **kwargs: 其余字段覆盖值。

        Returns:
            EvalConfig 实例。
        """
        # 1. 相对路径统一相对仓库根目录解析，避免"在哪个目录执行"影响结果
        from eval._bootstrap import REPO_ROOT

        csv_p = Path(csv_path)
        out_p = Path(out_dir)
        if not csv_p.is_absolute():
            csv_p = (REPO_ROOT / csv_p).resolve()
        if not out_p.is_absolute():
            out_p = (REPO_ROOT / out_p).resolve()

        # 2. 裁判模型默认读环境变量；没配就留空，由调用方决定回落到生成模型
        judge = kwargs.pop("judge_model", None)
        if judge is None:
            judge = os.getenv("EVAL_JUDGE_MODEL", "").strip()

        return cls(csv_path=csv_p, out_dir=out_p, judge_model=judge, **kwargs)

    @property
    def records_path(self) -> Path:
        """逐样本中间结果（可复用，避免重复跑图花钱）。"""
        return self.out_dir / "records.jsonl"