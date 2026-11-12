"""知识库概览服务：给前端"入库端"展示已经进了哪些文档/知识主体。

数据直接从 Milvus 的知识主体集合里聚合，不额外维护一份元数据表，
避免出现"库里删了、列表还在"的不一致。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.processor.import_process.config import get_config
from app.utils.milvus_util import get_milvus_client

logger = logging.getLogger(__name__)

# 一次最多扫描多少条主体记录（主体数量远小于切片数量，1000 足够演示与中小知识库）
_MAX_SCAN = 1000


def list_documents() -> List[Dict[str, Any]]:
    """列出已入库的知识主体。

    Returns:
        [{"item_name": 主体名, "file_title": 来源文档, "chunks": 该主体的切片数}]
        集合不存在或 Milvus 不可用时返回空列表（调用方按"空知识库"展示）。
    """
    client = get_milvus_client()
    if client is None:
        logger.warning("Milvus 不可用，返回空的知识库列表")
        return []

    collection = get_config().item_name_collection

    # 1. 集合还没建（一次都没导入过）属于正常状态，不当成错误
    try:
        if not client.has_collection(collection_name=collection):
            return []
    except Exception as e:
        logger.error(f"检查集合 {collection} 是否存在失败: {e}")
        return []

    # 2. 拉取主体记录；同名主体按 file_title 去重
    try:
        rows = client.query(
            collection_name=collection,
            filter="pk != ''",
            output_fields=["item_name", "file_title"],
            limit=_MAX_SCAN,
        )
    except Exception as e:
        logger.error(f"查询知识主体列表失败: {e}")
        return []

    seen = set()
    items: List[Dict[str, Any]] = []
    for row in rows or []:
        item_name = str(row.get("item_name") or "").strip()
        file_title = str(row.get("file_title") or "").strip()
        if not item_name:
            continue
        key = (item_name, file_title)
        if key in seen:
            continue
        seen.add(key)
        items.append({"item_name": item_name, "file_title": file_title})

    # 3. 顺带给每个主体统计切片数，前端展示"这篇文档贡献了多少块"
    chunk_counts = _count_chunks_by_item(client)
    for it in items:
        it["chunks"] = chunk_counts.get(it["item_name"], 0)

    items.sort(key=lambda x: (x["item_name"], x["file_title"]))
    return items


def _count_chunks_by_item(client) -> Dict[str, int]:
    """统计每个知识主体在切片集合里的条数；失败时返回空字典（不影响主列表）。"""
    collection = get_config().chunks_collection
    try:
        if not client.has_collection(collection_name=collection):
            return {}
        rows = client.query(
            collection_name=collection,
            filter="chunk_id > 0",
            output_fields=["item_name"],
            limit=10000,
        )
    except Exception as e:
        logger.warning(f"统计切片数量失败（忽略）: {e}")
        return {}

    counts: Dict[str, int] = {}
    for row in rows or []:
        name = str(row.get("item_name") or "").strip()
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts