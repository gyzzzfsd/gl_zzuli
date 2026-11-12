"""后端统一入口：一个 FastAPI 应用同时提供"入库端"和"问答端"接口。

原项目把导入和查询拆成两个独立应用、两个端口（8000/8001），
这里合并成一个进程，部署只要一条命令：`python run.py`。

启动方式：
    cd backend
    python run.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.import_router import register_import_routes
from app.api.query_router import register_query_routes
from app.core.settings import BACKEND_DIR, ensure_data_dirs, get_setting

logger = logging.getLogger("app.main")

# 前端构建产物目录：frontend/dist（`npm run build` 生成）。
# 存在时后端顺手把它托管在 8000 上，这样开发/演示只开一个端口就能看页面。
DIST_DIR: Path = BACKEND_DIR.parent / "frontend" / "dist"


def create_app() -> FastAPI:
    """创建并装配 FastAPI 应用。

    Returns:
        已注册全部路由与中间件的应用实例。
    """
    # 1. 基础信息（/docs 里能看到）
    app = FastAPI(
        title="Bio Knowledge Retrieval",
        description="生物知识检索系统：文档切片入库 + 多路检索问答",
        version="1.0.0",
    )

    # 2. CORS：只放行配置里列出的前端地址
    origins = [o.strip() for o in get_setting("CORS_ORIGINS", "").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["http://localhost:5173"],
        allow_credentials=False,   # 用 API Key 而不是 Cookie，避免 "*" + credentials 的非法组合
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 3. 接口鉴权：配置了 API_KEY 才启用
    #    EventSource 不能自定义请求头，所以 SSE 允许用 ?api_key= 传递
    api_key = get_setting("API_KEY", "")

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        # 3.1 未配置 Key：开发模式，直接放行（README 里已标注上线必须配置）
        if not api_key:
            return await call_next(request)

        # 3.2 健康检查与文档不拦，方便探活和调试
        path = request.url.path
        if path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # 3.3 依次从请求头、query 参数取 Key
        provided = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if provided != api_key:
            return JSONResponse(status_code=401, content={"detail": "缺少或错误的 API Key"})

        return await call_next(request)

    # 4. 健康检查：给部署探活用，顺带暴露版本
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "bio-knowledge-retrieval", "version": "1.0.0"}

    # 5. 两个端的路由
    register_import_routes(app)
    register_query_routes(app)

    # 6. 前端静态资源：只挂 /assets，不用 mount("/")，
    #    否则会把 /docs、/openapi.json 一起吃掉。
    assets_dir = DIST_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    # 7. 首页：有构建产物就返回页面，没有就给一句明确提示，
    #    免得直接在浏览器打开 8000 看到一个裸 404。
    @app.get("/", include_in_schema=False)
    async def index():
        index_file = DIST_DIR / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        return JSONResponse(
            {
                "service": "bio-knowledge-retrieval",
                "message": "后端已启动，但还没有前端构建产物。",
                "hint": "开发模式请另开终端在 frontend/ 下执行 npm run dev，访问 http://127.0.0.1:5173；"
                        "或用 npm run build 生成 dist 后重启后端，即可直接访问本端口。",
                "docs": "/docs",
                "health": "/health",
            }
        )

    return app


# uvicorn 也支持 `uvicorn app.main:app`
app = create_app()


def _prepare() -> None:
    """进程级初始化：建数据目录、配日志。"""
    ensure_data_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


_prepare()