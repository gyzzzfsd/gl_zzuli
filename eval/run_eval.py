"""评估命令行入口。

用法（从仓库根目录执行，任意 cwd 均可）：
    # 0. 先做数据集体检，不跑图、不花钱
    python eval/run_eval.py --dry-run

    # 1. 跑 6 条样例，只要确定性指标（不需要 ragas）
    python eval/run_eval.py --csv eval/eval_doc/qa_sample.csv --limit 6

    # 2. 全量 + RAGAS 五项（需要先 pip install -r requirements-eval.txt）
    python eval/run_eval.py --csv eval/eval_doc/qa.csv --mode all

    # 3. 只重跑 RAGAS（复用上次的 records.jsonl，不重复调检索/生成）
    python eval/run_eval.py --csv eval/eval_doc/qa.csv --mode ragas --records eval/reports/xxx/records.jsonl

    # 4. 把某次结果固化为基线；之后每次跑都会与它比对
    python eval/run_eval.py --save-baseline eval/reports/20250101-120000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 仓库根目录入 sys.path，保证 `python eval/run_eval.py` 与 `python -m eval.run_eval` 都能跑
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval._bootstrap import ensure_backend_on_path  # noqa: E402
from eval.config import EvalConfig  # noqa: E402
from eval.dataset import audit_qa, load_qa, stratified_counts  # noqa: E402


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="生物医学知识库 RAG 评估")
    parser.add_argument("--csv", default="eval/eval_doc/qa_sample.csv", help="黄金集路径")
    parser.add_argument("--out", default="", help="输出目录；留空则按时间戳创建")
    parser.add_argument("--mode", choices=("ext", "ragas", "all", "report"), default="ext",
                        help="ext=只跑确定性指标 / ragas=只跑 RAGAS / all=全跑 / report=只重出报告（读 records.jsonl）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 表示全量）")
    parser.add_argument("--batch", type=int, default=20, help="RAGAS 分批大小")
    parser.add_argument("--judge-model", default="", help="裁判模型名；留空沿用生成模型")
    parser.add_argument("--records", default="", help="已有的 records.jsonl（mode=ragas 时复用）")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已完成的样本")
    parser.add_argument("--dry-run", action="store_true", help="只校验数据集，不跑图")
    parser.add_argument("--no-gate", action="store_true", help="不做基线比对（只看分数）")
    parser.add_argument("--save-baseline", default="", help="把指定报告目录固化为基线后退出")
    parser.add_argument("--baseline", default="eval/baseline.json", help="基线文件路径")
    return parser.parse_args(argv)


def _load_records(path: Path) -> List[Dict[str, Any]]:
    """从 JSONL 读取已有记录。"""
    if not path.is_file():
        raise FileNotFoundError(f"未找到记录文件 {path}；先跑一次 --mode ext 或 --mode all")
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _latest_records() -> Optional[Path]:
    """找最近一次评估的 records.jsonl（report 模式下省略 --records 时用）。

    Returns:
        路径；一个都没有时返回 None。
    """
    root = REPO_ROOT / "eval" / "reports"
    candidates = sorted(root.glob("*/records.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _print_dataset_overview(items, warnings: List[str]) -> None:
    """打印数据集体检结果。"""
    counts = stratified_counts(items)
    print(f"样本数：{len(items)}")
    print("题型分布：" + "，".join(f"{k}={v}" for k, v in counts.items()))
    if warnings:
        print(f"⚠ 体检告警 {len(warnings)} 条：")
        for line in warnings[:10]:
            print(f"  - {line}")
        if len(warnings) > 10:
            print(f"  … 其余 {len(warnings) - 10} 条省略")
    else:
        print("体检：未发现问题")


def main(argv: Optional[List[str]] = None) -> int:
    """主流程。

    Returns:
        进程退出码：0 成功；1 门禁未通过；2 数据/环境问题。
    """
    args = _parse_args(argv)

    # 1. 固化基线是独立动作，不需要跑评估
    if args.save_baseline:
        from eval.gate import save_baseline

        target = save_baseline(Path(args.save_baseline), Path(args.baseline))
        print(f"基线已保存：{target}")
        return 0

    # 2. 组装配置（时间戳目录，避免覆盖历史报告）
    out_dir = args.out or f"eval/reports/{datetime.now():%Y%m%d-%H%M%S}"
    cfg = EvalConfig.from_args(
        csv_path=args.csv,
        out_dir=out_dir,
        limit=args.limit or None,
        mode=args.mode,
        ragas_batch_size=args.batch,
        judge_model=args.judge_model,
    )

    # 3. 加载数据集（失败直接退出，别浪费时间跑图）
    try:
        items = load_qa(cfg.csv_path, cfg.limit)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[数据错误] {exc}")
        return 2

    warnings = audit_qa(items)
    _print_dataset_overview(items, warnings)

    if args.dry_run:
        print("dry-run：只校验数据，不跑图。")
        return 0

    if args.limit and len(items) > args.limit:
        items = items[:args.limit]

    # 4. 跑图（ext/all 模式）
    records: List[Dict[str, Any]]
    if args.mode in ("ext", "all"):
        from eval.runner import BackendUnavailable, run_samples

        try:
            ensure_backend_on_path()
        except FileNotFoundError as exc:
            print(f"[环境错误] {exc}")
            return 2

        print(f"\n开始跑图：{len(items)} 条 → {cfg.records_path}")
        try:
            records = run_samples(items, cfg.records_path, resume=args.resume)
        except BackendUnavailable as exc:
            print(f"[环境错误] {exc}")
            return 2
    else:
        # ragas / report 模式：复用已有记录，不再跑图（省钱、也便于离线重出报告）
        if args.records:
            records_path = Path(args.records)
        elif args.mode == "report":
            records_path = _latest_records() or cfg.records_path
        else:
            records_path = cfg.records_path
        try:
            records = _load_records(records_path)
        except FileNotFoundError as exc:
            print(f"[数据错误] {exc}")
            return 2
        print(f"复用已有记录：{records_path}（{len(records)} 条）")

    # 5. RAGAS（可选）
    ragas_result: Dict[str, Any] = {}
    if args.mode in ("ragas", "all"):
        from eval.ragas_adapter import RagasUnavailable, evaluate_records

        try:
            print(f"\n开始 RAGAS 评估：{len(records)} 条，batch={cfg.ragas_batch_size}")
            ragas_result = evaluate_records(records, cfg)
        except RagasUnavailable as exc:
            print(f"[依赖缺失] {exc}")
            return 2
        except Exception as exc:  # noqa: BLE001 - 裁判链路可能整体失败，别吞掉原因
            print(f"[RAGAS 失败] {type(exc).__name__}: {exc}")
            return 2
        if ragas_result.get("failures"):
            print(f"⚠ {len(ragas_result['failures'])} 批评估失败，分数按 NaN 处理")

    # 6. 汇总与报告
    from eval import report as report_mod

    items_by_qid = {item.qid: item for item in items}
    rows = report_mod.enrich(records, items_by_qid, ragas_result.get("per_sample"))
    snapshot = _snapshot_safely()
    summary = report_mod.summarize(rows, cfg, args.mode, snapshot)

    baseline: Optional[Dict[str, Any]] = None
    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = REPO_ROOT / baseline_path
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        summary["baseline_file"] = str(baseline_path.relative_to(REPO_ROOT))

    paths = report_mod.write_all(rows, summary, cfg, baseline)
    print("\n报告产物：")
    for name, path in paths.items():
        print(f"  {name}: {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")

    # 7. 打印总体指标（一眼看结果）
    print("\n总体指标：")
    for name, value in (summary.get("overall") or {}).items():
        flag = "" if value is None else ("  ✗" if value < cfg.thresholds.get(name, -1) else "  ✓")
        print(f"  {name:<20}{_fmt(value)}{flag}")

    # 8. 门禁
    if args.no_gate:
        print("\n已跳过门禁（--no-gate）")
        return 0

    from eval.gate import check

    failures = check(summary, baseline, cfg.thresholds, cfg.regression_tolerance)
    if failures:
        print("\n门禁未通过：")
        for reason in failures:
            print(f"  ✗ {reason}")
        return 1

    print("\n门禁通过。")
    return 0


def _fmt(value: Optional[float]) -> str:
    """数值格式化。"""
    return "—" if value is None else f"{value:.4f}"


def _snapshot_safely() -> Dict[str, Any]:
    """采集参数快照；失败返回空字典（不能因为快照失败把评估结果弄丢）。"""
    try:
        from eval.runner import collect_config_snapshot

        return collect_config_snapshot()
    except Exception as exc:  # noqa: BLE001
        print(f"⚠ 参数快照采集失败（不影响评估）：{exc}")
        return {}


if __name__ == "__main__":
    raise SystemExit(main())