"""回归门禁：把"指标退化"变成 CI 能拦住的红灯。

两条判据，任一不满足即失败：
1. 绝对门槛：本次指标低于 config.DEFAULT_THRESHOLDS；
2. 相对基线：本次比基线低超过容忍度（默认 0.02）。

为什么用"双判据"：只看绝对门槛会让"从 0.95 掉到 0.86"悄悄过关；
只看相对基线又会让"一直很差的指标"永远不报警。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _fmt(value: Optional[float]) -> str:
    """数值格式化，None 显示为 —。"""
    return "—" if value is None else f"{value:.4f}"


def check(summary: Dict[str, Any], baseline: Optional[Dict[str, Any]],
          thresholds: Dict[str, float], tolerance: float) -> List[str]:
    """执行门禁检查。

    Args:
        summary: 本次评估的 summary（report.summarize 的产物）。
        baseline: 基线 summary；None 表示只做绝对门槛检查。
        thresholds: 指标门槛。
        tolerance: 相对基线的允许跌幅。

    Returns:
        失败原因列表；空列表表示通过。
    """
    failures: List[str] = []
    overall = summary.get("overall") or {}
    base_overall = (baseline or {}).get("overall") or {}

    for name, value in overall.items():
        # 1. 绝对门槛
        threshold = thresholds.get(name)
        if threshold is not None and value is not None and value < threshold:
            failures.append(f"`{name}`={_fmt(value)} 低于门槛 {threshold:.2f}")

        # 2. 相对基线
        base = base_overall.get(name)
        if value is not None and base is not None and value < base - tolerance:
            failures.append(f"`{name}`={_fmt(value)} 较基线 {_fmt(base)} 下降超过 {tolerance:.2f}")

        # 3. 指标消失（本次没算出来）也要拦：NaN 往往是硬故障
        if value is None:
            failures.append(f"`{name}` 本次未计算出（NaN），检查该批样本或裁判调用")

    # 4. 执行期错误
    if summary.get("error_count"):
        failures.append(f"有 {summary['error_count']} 条样本执行报错")

    return failures


def save_baseline(report_dir: Path, baseline_path: Path) -> Path:
    """把某次报告固化为基线。

    Args:
        report_dir: 报告目录（含 summary.json）。
        baseline_path: 基线文件输出路径。

    Returns:
        基线文件路径。

    Raises:
        FileNotFoundError: 报告目录里没有 summary.json。
    """
    source = report_dir / "summary.json"
    if not source.is_file():
        raise FileNotFoundError(f"未找到 {source}，请先跑一次评估")
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, baseline_path)
    return baseline_path