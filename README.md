# Bio Knowledge Retrieval

面向生物医学文献的知识检索系统：把 PDF / Markdown 文档切片入库，构建向量索引与知识图谱，
然后通过多路召回 + 重排回答问题。前后端分离，后端一个进程，前端两个页面（入库端 / 问答端）。

---

## 一、能力概览

**入库端（Import）**

```
上传文件 → 文件类型判断 → PDF 转 Markdown(MinerU) / 直接读 md
        → 图片摘要与 MinIO 托管 → 按标题切片 → 知识主体识别
        → BGE-M3 稠密+稀疏向量 → 写入 Milvus
        → LLM 抽取实体关系 → 写入 Neo4j + 实体向量库
```

- 切片策略：按 1–6 级标题切分，短块合并、长块二次切分，保留 `title / parent_title / file_title`
- 知识主体（`item_name`）：一份文档真正在讲的那个生物对象（基因 / 物种 / 通路 / 疾病 / 方法 / 试剂）
- 幂等：同名知识主体重复导入时，Neo4j 里该主体的旧子图会被覆盖重建

**问答端（Chat）**

```
问题 → 知识主体识别 + 指代消解 + 查询改写
     → 四路并行检索：切片向量 / HyDE 假设文档 / 知识图谱 / 联网(可选)
     → RRF 融合 → bge-reranker 精排 + 分数断崖截断
     → 拼装上下文（文档 + 历史 + 图谱三元组）→ LLM 流式生成
```

- 会话级记忆：最近 10 条消息存 MongoDB，用于多轮指代消解
- 流式：SSE 推送 `ready / progress / delta / final`
- 答案后处理：模型编造的图片链接会被白名单校验剔除

---

## 二、目录结构

```
bio_knowledge_retrieval/
├── backend/
│   ├── app/
│   │   ├── main.py                  # 统一 FastAPI 入口（入库 + 问答）
│   │   ├── api/                     # 两组路由
│   │   ├── core/                    # settings（.env 唯一入口）、paths、deps
│   │   ├── processor/
│   │   │   ├── import_process/      # 导入 LangGraph（8 节点）
│   │   │   └── query_process/       # 查询 LangGraph（9 节点 + 并行超级步）
│   │   ├── prompts/                 # 查询提示词
│   │   ├── service/                 # 导入/查询/任务/知识库概览
│   │   ├── utils/                   # Milvus / Neo4j / Mongo / MinIO / BGE / SSE
│   │   └── schema/                  # Pydantic 模型
│   ├── samples/                     # 可直接导入的示例文档
│   ├── data/                        # 运行期产物（自动创建，已 gitignore）
│   ├── requirements.txt
│   ├── .env.example                 # 环境变量模板
│   └── run.py                       # 启动入口：python run.py
└── frontend/
    ├── src/views/IngestView.vue     # 入库端
    ├── src/views/ChatView.vue       # 问答端
    ├── src/api.js                   # 接口封装（含 SSE）
    ├── vite.config.js               # /api 代理到后端
    └── package.json
```

---

## 三、依赖服务

| 组件 | 用途 | 是否必需 |
|---|---|---|
| Milvus | 切片向量 + 知识主体名 + 实体名 | 必需 |
| Neo4j | 实体关系图谱 | 必需 |
| MongoDB | 会话历史（短期记忆） | 必需 |
| MinIO | 原始文件与解析图片的托管 | 导入含图片的 PDF 时必需 |
| LLM（OpenAI 兼容） | 主体识别 / 图谱抽取 / 答案生成 | 必需 |
| BGE-M3 + bge-reranker | 本地嵌入与重排（GPU 可选） | 必需 |
| MinerU | PDF → Markdown | 仅导入 PDF 时必需；`.md` 导入不需要 |

---

## 四、接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/import/upload` | 上传文档（multipart），返回 `task_id` |
| GET | `/api/import/status/{task_id}` | 任务状态 + 节点进度 + 失败原因 |
| GET | `/api/import/documents` | 已入库知识主体列表 |
| POST | `/api/query` | 提问；`is_stream=true` 返回 `task_id`，否则直接返答案 |
| GET | `/api/query/stream/{task_id}` | SSE：`ready` / `progress` / `delta` / `final` |
| GET | `/api/session/new` | 生成会话 ID |
| GET | `/api/history/{session_id}` | 会话历史 |
| DELETE | `/api/history/{session_id}` | 清空会话历史 |
| GET | `/health` | 健康检查 |

