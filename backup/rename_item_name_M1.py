"""把 M1 的 item_name 从"回退成文件标题"改成规范产品名。

背景：M1 导入时 LLM 没能识别出主体名，代码按安全策略回退成了 file_title
（`家用上臂式电子血压计M1使用手册`），于是 Milvus 的 chunks / subject_names /
entity_names 三个集合和 Neo4j 的实体节点里都存了这个不规范的名字。
本脚本把四处一起改掉，并先做完整备份，可用 restore 模式回滚。

用法：
    python rename_item_name_M1.py backup    # 只备份
    python rename_item_name_M1.py rename    # 备份 + 改名 + 自检
    python rename_item_name_M1.py restore   # 从备份回滚（改名前状态）

注意：改名必须重算主体名向量（subject_names 的向量就是主体名的嵌入），
否则 item_name_confirm 的主体对齐会立刻失效。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2] / "bio_knowledge_retrieval" / "backend"
if not BACKEND.exists():
    BACKEND = Path(r"D:\allCode\AI_Model\project\Project_1_Knowledge_Base\bio_knowledge_retrieval\backend")
sys.path.insert(0, str(BACKEND))

from app.utils.bge_m3_embedding_util import get_beg_m3_embedding_model  # noqa: E402
from app.utils.milvus_util import get_milvus_client  # noqa: E402
from app.utils.neo4j_util import get_neo4j_driver  # noqa: E402

OLD_NAME = "家用上臂式电子血压计M1使用手册"
NEW_NAME = "家用上臂式电子血压计 M1"
CHUNKS = "bio_chunks_v1"
SUBJECTS = "bio_subject_names_v1"
ENTITIES = "bio_entity_names_v1"

BACKUP_PATH = Path(__file__).resolve().parent / "item_name_M1_backup.json"


def _dump(rows: list[dict]) -> list[dict]:
    """把稀疏向量转成 [[token_id, weight], ...]，避免 JSON 把整数键变成字符串。"""
    out = []
    for row in rows:
        item = dict(row)
        sparse = item.get("sparse_vector")
        if isinstance(sparse, dict):
            item["sparse_vector"] = [[int(k), float(v)] for k, v in sparse.items()]
        out.append(item)
    return out


def _load_sparse(pairs):
    return {int(k): float(v) for k, v in pairs}


def do_backup() -> dict:
    client = get_milvus_client()
    chunks = client.query(CHUNKS, filter=f'item_name == "{OLD_NAME}"', output_fields=["*"], limit=2000)
    ents = client.query(ENTITIES, filter=f'item_name == "{OLD_NAME}"', output_fields=["*"], limit=2000)
    subs = client.query(SUBJECTS, filter=f'item_name == "{OLD_NAME}"', output_fields=["*"], limit=100)

    neo_rows = []
    driver = get_neo4j_driver()
    if driver is not None:
        with driver.session() as session:
            for rec in session.run(
                "MATCH (n) WHERE n.item_name = $old "
                "RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props",
                old=OLD_NAME,
            ):
                neo_rows.append({"eid": rec["eid"], "labels": list(rec["labels"]), "props": rec["props"]})

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "old_name": OLD_NAME,
        "new_name": NEW_NAME,
        "milvus_chunks": _dump(chunks),
        "milvus_entities": _dump(ents),
        "milvus_subjects": _dump(subs),
        "neo4j_nodes": neo_rows,
    }
    BACKUP_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"备份完成：{BACKUP_PATH}")
    print(f"  chunks {len(chunks)} 条 | entities {len(ents)} 条 | subjects {len(subs)} 条 | neo4j 节点 {len(neo_rows)} 个")
    return payload


def do_rename() -> None:
    client = get_milvus_client()

    # 1. 重算主体名向量（subject_names 的向量必须是名字的嵌入）
    model = get_beg_m3_embedding_model()
    res = model.encode_documents([NEW_NAME])
    dense = res["dense"][0].tolist()
    start, end = res["sparse"].indptr[0], res["sparse"].indptr[1]
    sparse = {
        int(i): float(w)
        for i, w in zip(res["sparse"].indices[start:end], res["sparse"].data[start:end])
    }

    # 2. 主体集合：先插新行再删旧行（pk 是 auto_id，不能 upsert）
    old_subs = client.query(SUBJECTS, filter=f'item_name == "{OLD_NAME}"', output_fields=["pk", "file_title"], limit=10)
    if old_subs:
        file_title = old_subs[0].get("file_title") or ""
        client.insert(
            collection_name=SUBJECTS,
            data=[{"file_title": file_title, "item_name": NEW_NAME,
                   "dense_vector": dense, "sparse_vector": sparse}],
        )
        client.delete(collection_name=SUBJECTS, ids=[s["pk"] for s in old_subs])
        print(f"主体集合：插入新行 1 条，删除旧行 {len(old_subs)} 条")
    else:
        print("主体集合：未找到旧行，跳过")

    # 3. 切片集合：整行回写，只改 item_name
    chunks = client.query(CHUNKS, filter=f'item_name == "{OLD_NAME}"', output_fields=["*"], limit=2000)
    for row in chunks:
        row["item_name"] = NEW_NAME
    if chunks:
        client.upsert(collection_name=CHUNKS, data=chunks)
    print(f"切片集合：回写 {len(chunks)} 条")

    # 4. 实体集合：同理（单次回写行数多，按 100 条一批，避免单请求过大）
    ents = client.query(ENTITIES, filter=f'item_name == "{OLD_NAME}"', output_fields=["*"], limit=2000)
    for row in ents:
        row["item_name"] = NEW_NAME
    for i in range(0, len(ents), 100):
        client.upsert(collection_name=ENTITIES, data=ents[i:i + 100])
    print(f"实体集合：回写 {len(ents)} 条")

    # 5. Neo4j：同一属性批量改写
    driver = get_neo4j_driver()
    if driver is not None:
        with driver.session() as session:
            rec = session.run(
                "MATCH (n) WHERE n.item_name = $old SET n.item_name = $new RETURN count(n) AS c",
                old=OLD_NAME, new=NEW_NAME,
            ).single()
            print(f"Neo4j：改写节点 {rec['c']} 个")


def do_restore() -> None:
    payload = json.loads(BACKUP_PATH.read_text(encoding="utf-8"))
    client = get_milvus_client()

    # 1. 切片与实体：回写旧名字
    for coll, key in ((CHUNKS, "milvus_chunks"), (ENTITIES, "milvus_entities")):
        rows = payload[key]
        for row in rows:
            row["sparse_vector"] = _load_sparse(row["sparse_vector"])
        # 先删掉新名字的行，避免同一 chunk_id 被 upsert 误留旧内容
        new_rows = client.query(coll, filter=f'item_name == "{NEW_NAME}"', output_fields=["*"], limit=2000)
        for row in new_rows:
            row["item_name"] = OLD_NAME
        if new_rows:
            client.upsert(collection_name=coll, data=new_rows)
        print(f"{coll}：回滚 {len(new_rows)} 条")

    # 2. 主体集合：删除新行，重建旧行
    client.delete(collection_name=SUBJECTS, filter=f'item_name == "{NEW_NAME}"')
    subs = payload["milvus_subjects"]
    for row in subs:
        row["sparse_vector"] = _load_sparse(row["sparse_vector"])
    if subs:
        client.insert(collection_name=SUBJECTS, data=subs)
    print(f"主体集合：回滚 {len(subs)} 条")

    # 3. Neo4j
    driver = get_neo4j_driver()
    if driver is not None and payload["neo4j_nodes"]:
        with driver.session() as session:
            rec = session.run(
                "MATCH (n) WHERE n.item_name = $new SET n.item_name = $old RETURN count(n) AS c",
                old=OLD_NAME, new=NEW_NAME,
            ).single()
            print(f"Neo4j：回滚 {rec['c']} 个节点")


def verify() -> None:
    client = get_milvus_client()
    for coll in (CHUNKS, SUBJECTS, ENTITIES):
        old = len(client.query(coll, filter=f'item_name == "{OLD_NAME}"', output_fields=["item_name"], limit=1000) or [])
        new = len(client.query(coll, filter=f'item_name == "{NEW_NAME}"', output_fields=["item_name"], limit=1000) or [])
        print(f"  {coll:<22} 旧名 {old} 条 / 新名 {new} 条")
    driver = get_neo4j_driver()
    if driver is not None:
        with driver.session() as session:
            old = session.run("MATCH (n) WHERE n.item_name = $o RETURN count(n) AS c", o=OLD_NAME).single()["c"]
            new = session.run("MATCH (n) WHERE n.item_name = $n RETURN count(n) AS c", n=NEW_NAME).single()["c"]
        print(f"  Neo4j 节点            旧名 {old} 个 / 新名 {new} 个")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "rename"
    if action == "backup":
        do_backup()
    elif action == "rename":
        do_backup()
        do_rename()
        print("\n自检：")
        verify()
    elif action == "restore":
        do_restore()
        print("\n自检：")
        verify()
    else:
        raise SystemExit(f"未知动作 {action}，可用：backup / rename / restore")