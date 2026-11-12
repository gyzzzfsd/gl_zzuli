"""问答端接口：提交问题（同步/流式）、SSE 推送、会话历史。"""

from __future__ import annotations

from fastapi import BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.core.deps import get_query_service
from app.schema.query_schema import QueryRequest, QueryResponse, StreamSubmitResponse
from app.service.query_service import QueryService
from app.utils.sse_util import sse_generator


def register_query_routes(app) -> None:
    """把问答端路由注册到传入的 FastAPI 实例上。

    Args:
        app: FastAPI 应用实例（统一入口 app/main.py 创建）。
    """

    @app.get("/api/session/new")
    async def new_session(service: QueryService = Depends(get_query_service)):
        """生成一个新会话 ID（前端在首次问答前调用）。"""
        return {"session_id": service.generate_session_id()}

    @app.post("/api/query")
    async def query(
        request: QueryRequest,
        background_tasks: BackgroundTasks,
        service: QueryService = Depends(get_query_service),
    ):
        """提交问题。

        - is_stream=False：同步返回完整答案（适合 curl 调试）。
        - is_stream=True：立即返回 task_id，前端再连 SSE 拿增量。
        """
        session_id = request.session_id or service.generate_session_id()
        task_id = service.generate_task_id()
        service.submit_query(task_id, request.is_stream)

        # 1. 流式：丢后台，立刻返回任务号
        if request.is_stream:
            background_tasks.add_task(
                service.run_query_graph, task_id, session_id, request.query, True
            )
            return StreamSubmitResponse(
                message="Query submitted", session_id=session_id, task_id=task_id
            )

        # 2. 非流式：整条图跑完再返回。放线程池里跑，避免阻塞事件循环
        await run_in_threadpool(
            service.run_query_graph, task_id, session_id, request.query, False
        )
        answer = service.get_answer(task_id)
        return QueryResponse(message="处理完成", session_id=session_id, answer=answer)

    @app.get("/api/query/stream/{task_id}")
    async def stream(task_id: str, request: Request):
        """SSE 推送：progress / delta / final 三类事件。"""
        return StreamingResponse(
            sse_generator(task_id, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/history/{session_id}")
    async def get_history(
        session_id: str,
        limit: int = 50,
        service: QueryService = Depends(get_query_service),
    ):
        """读取某个会话的历史消息（按时间正序）。"""
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取历史失败: {e}")

    @app.delete("/api/history/{session_id}")
    async def clear_chat_history(
        session_id: str,
        service: QueryService = Depends(get_query_service),
    ):
        """清空某个会话的历史消息。"""
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}