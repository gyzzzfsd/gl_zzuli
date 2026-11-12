"""生物知识检索系统 —— 后端唯一启动入口。

在 PyCharm 里：右键本文件 → Run 'run'，即可启动整个后端（无需额外 Run Configuration）。
命令行等价写法：`python run.py`。

为了让"右键直接 Run"万无一失，这里做了三件事：
1. 把本文件所在目录（backend/）加入 sys.path —— 否则 `import app` 会失败；
2. 把进程工作目录切到 backend/ —— PyCharm 默认工作目录取决于项目设置，
   切一下可以保证 .env、data/、模型缓存等相对路径都落在正确位置；
3. 单 worker 运行 —— 任务状态与 SSE 队列都在进程内存里（见 app/utils/task_util.py），
   多 worker 会让流式接口打到没有队列的实例上。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 目录层级：backend/run.py -> 本文件所在目录就是 backend
BACKEND_DIR = Path(__file__).resolve().parent

# 1. 保证任意工作目录下都能 import app 包
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# 2. 固定工作目录：settings 用绝对路径找 .env，但 MinerU 与部分工具按 cwd 解析相对路径
os.chdir(BACKEND_DIR)

import uvicorn  # noqa: E402  （必须在 sys.path 调整之后导入）

from app.core.settings import ENV_FILE, get_int, get_setting  # noqa: E402
from app.main import app  # noqa: E402


def main() -> None:
    """启动 uvicorn。端口与监听地址来自 .env 的 APP_PORT / APP_HOST。"""
    # 启动自检：.env 缺失是最常见的"起来了但连不上库"的原因，提前说清楚
    if not ENV_FILE.exists():
        print(f"[bio-kb] 警告：未找到 {ENV_FILE}，请先复制 .env.example 为 .env")

    host = get_setting("APP_HOST", "0.0.0.0")
    port = get_int("APP_PORT", 8000)

    print(f"[bio-kb] 配置：{ENV_FILE}")
    print(f"[bio-kb] 后端启动：http://127.0.0.1:{port}  (API 文档 /docs)")
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()