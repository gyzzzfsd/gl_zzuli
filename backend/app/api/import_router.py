"""导入端接口：上传文档 → 后台跑 LangGraph 导入流水线 → 轮询进度。"""

from __future__ import annotations

import io
from typing import Any, Dict, List

from fastapi import BackgroundTasks, Depends, File, HTTPException, UploadFile

from app.core.deps import get_import_file_service, get_task_service
from app.core.settings import get_int, get_setting
from app.schema.task_schema import TaskStatusResponse
from app.schema.upload_schema import UploadResponse
from app.service.import_file_service import ImportFileService
from app.service.kb_service import list_documents
from app.service.task_service import TaskService

# 允许上传的文档类型：PDF 走 MinerU 解析，Markdown 直接进入清洗/切分
ALLOWED_SUFFIXES = {".pdf", ".md"}


def register_import_routes(app) -> None:
    """把导入端路由注册到传入的 FastAPI 实例上。

    Args:
        app: FastAPI 应用实例（统一入口 app/main.py 创建）。
    """

    @app.post("/api/import/upload", response_model=UploadResponse)
    async def upload_file_endpoint(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        service: ImportFileService = Depends(get_import_file_service),
    ):
        """上传文档并异步启动入库流程。

        Returns:
            task_id：前端拿它轮询 /api/import/status/{task_id}。
        """
        # 1. 校验文件名与扩展名（前端也校验，但后端不能只信前端）
        filename = (file.filename or "").strip()
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"只支持 {', '.join(sorted(ALLOWED_SUFFIXES))} 格式，收到：{suffix or '未知'}",
            )

        # 2. 读取内容并校验大小（读进来再判断，避免半截文件落盘）
        max_bytes = get_int("MAX_UPLOAD_MB", 50) * 1024 * 1024
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="文件内容为空")
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"文件超过 {get_setting('MAX_UPLOAD_MB', '50')} MB 限制",
            )
        # 交给 service 的仍是 UploadFile 接口，用 BytesIO 包一层即可
        file.file = io.BytesIO(content)

        # 3. 落盘（本地 + MinIO），生成 task_id
        task_id, file_dir, import_file_path = service.process_upload_file(file)

        # 4. 后台跑导入流水线：请求立即返回，前端轮询进度
        background_tasks.add_task(service.run_import_graph, task_id, file_dir, import_file_path)

        return UploadResponse(message="文件已接收，正在后台解析入库", task_id=task_id)

    @app.get("/api/import/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status_endpoint(
        task_id: str,
        task_service: TaskService = Depends(get_task_service),
    ):
        """查询某个导入任务的状态与节点进度。"""
        task_info = task_service.get_task_info(task_id)
        return TaskStatusResponse(**task_info)

    @app.get("/api/import/documents")
    def get_documents() -> Dict[str, List[Dict[str, Any]]]:
        """列出已入库的知识主体，用于前端展示知识库现状。

        用同步 def 而非 async：内部是 pymilvus 的阻塞查询，
        交给 FastAPI 的线程池执行，避免卡住事件循环。
        """
        try:
            return {"items": list_documents()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取知识库列表失败: {e}")