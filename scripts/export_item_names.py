"""导出库内 item_name，用于回填/校验黄金集的 item_name 列。

为什么需要：`item_name` 是检索的硬过滤键，CSV 里写错一个字，整条样本就会"检索不到"，
最后表现为 context_recall 掉到 0——看起来像召回差，其实是标注错。

用法：
    python scripts/export_item_names.py                       # 打印已入库主体
    python scripts/export_item_names.py --out item_names.csv  # 导出 CSV 备查
    python scripts/export_item_names.py --check eval/eval_doc/qa.csv  # 校验黄金集
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# 仓库根目录与 backend 都要在 sys.path 上（脚本在 scripts/ 下）
REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if path not in sys.path:
        sys.path.insert(0, path)


def fetch_item_names() -> list:
    """从 Milvus 主体集合读已入库的知识主体。

    Returns:
        [{"item_name": str, "file_title": str, "chunks": int}]

    Raises:
        SystemExit: Milvus 不可用或集合为空时给出明确提示。
    """
    from app.service.kb_service import list_documents

    documents = list_documents()
    if not documents:
        print("[错误] 未读到任何知识主体：Milvus 不可用，或还没有导入过文档")
        raise SystemExit(2)
    return documents


def print_table(documents: list) -> None:
    """打印主体清单（按 item_name 排序）。"""
    width = max(len(d["item_name"]) for d in documents)
    print(f"{'item_name'.ljust(width)}  {'chunks':>6}  file_title")
    print("-" * (width + 40))
    for doc in documents:
        print(f"{doc['item_name'].ljust(width)}  {doc.get('chunks', 0):>6}  {doc.get('file_title', '')}")
    print(f"\n共 {len(documents)} 条主体记录")


def write_csv(documents: list, out_path: Path) -> None:
    """导出为 CSV。

    Args:
        documents: 主体列表。
        out_path: 输出路径。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["item_name", "file_title", "chunks"])
        writer.writeheader()
        writer.writerows(documents)
    print(f"已导出：{out_path}")


def check_qa(items: list) -> int:
    """校验黄金集里的 item_name 是否都真实存在。

    比较采用"完全相等"，因为检索侧就是精确匹配（见 vector_search_node 的 expr 构造）。
    存在的但不完全一致（多空格、少后缀）也会被判缺失，这正是要暴露的问题。

    Args:
        items: [{"item_name": str, "doc": str, "question": str}]

    Returns:
        进程退出码：0 全部命中；1 有缺失。
    """
    existing = {str(doc["item_name"]).strip() for doc in items}
    missing = [row for row in items if str(row.get("item_name", "")).strip() not in existing]

    if not missing:
        print(f"✓ 黄金集 {len(items)} 条的 item_name 全部存在于库内")
        return 0

    print(f"✗ {len(missing)} 条样本的 item_name 在库内找不到：")
    for row in missing:
        print(f"  - {row.get('item_name', '')!r}（问题：{str(row.get('question', ''))[:40]}…）")
    print("\n提示：跑 `python scripts/export_item_names.py` 查看库内真实名称后回填。")
    return 1


def load_qa_rows(csv_path: Path) -> list:
    """读取黄金集（只取 item_name / question 两列，不校验其他列）。

    Args:
        csv_path: CSV 路径。

    Returns:
        行字典列表。

    Raises:
        SystemExit: 文件不存在或没有 item_name 列。
    """
    if not csv_path.is_file():
        print(f"[错误] 找不到 {csv_path}")
        raise SystemExit(2)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if "item_name" not in (reader.fieldnames or []):
            print("[错误] CSV 缺少 item_name 列")
            raise SystemExit(2)
        return list(reader)


def main(argv=None) -> int:
    """命令行入口。

    Returns:
        进程退出码。
    """
    parser = argparse.ArgumentParser(description="导出/校验知识主体名")
    parser.add_argument("--out", default="", help="导出 CSV 路径")
    parser.add_argument("--check", default="", help="校验指定黄金集 CSV 的 item_name")
    args = parser.parse_args(argv)

    documents = fetch_item_names()

    # 1. 校验模式：只比对，不打印全表（避免刷屏）
    if args.check:
        return check_qa(load_qa_rows(Path(args.check)))

    # 2. 默认打印
    print_table(documents)

    # 3. 导出
    if args.out:
        write_csv(documents, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())