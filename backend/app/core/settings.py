"""全局设置与目录约定。

本模块是**唯一**读取 `.env` 的地方：其他模块只通过 `get_setting()` 取值。
之所以集中在这里，是因为原项目里每个工具模块各自 `load_dotenv()`，
导致"到底读的是哪个 .env"取决于进程工作目录，换台机器就跑不起来。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 目录层级：backend/app/core/settings.py  ->  parents[0]=core, [1]=app, [2]=backend
BACKEND_DIR: Path = Path(__file__).resolve().parents[2]

ENV_FILE: Path = BACKEND_DIR / ".env"

# 显式指定路径加载：不依赖启动时的工作目录。
# override=False 表示已存在的系统环境变量优先，方便容器/CI 注入。
load_dotenv(dotenv_path=ENV_FILE, override=False)

# ---------------- 数据目录 ----------------
DATA_DIR: Path = BACKEND_DIR / "data"
UPLOAD_DIR: Path = DATA_DIR / "uploads"   # 原始上传文件（按 日期/task_id 归档）
PARSED_DIR: Path = DATA_DIR / "parsed"    # MinerU 解析产物、chunks.json 等中间文件
LOG_DIR: Path = DATA_DIR / "logs"


def ensure_data_dirs() -> None:
    """创建运行期需要的目录（幂等）。"""
    for d in (DATA_DIR, UPLOAD_DIR, PARSED_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------- 取值辅助函数 ----------------
def get_setting(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """读取字符串配置。

    Args:
        key: 环境变量名。
        default: 未配置时的默认值。
        required: 为 True 时，缺失直接抛错并指出缺哪个键，避免"空字符串静默连错库"。

    Returns:
        配置值（字符串）。

    Raises:
        RuntimeError: 配置了 required 但该键不存在。
    """
    value = os.getenv(key)

    # 1. 显式要求必填：尽早失败，把缺的键名写进异常
    if required and (value is None or value.strip() == ""):
        raise RuntimeError(
            f"缺少必填环境变量 {key}，请检查 {ENV_FILE}（可参考 .env.example）"
        )

    # 2. 未配置且无默认值：返回空串，由调用方决定是降级还是报错
    if value is None or value == "":
        return default if default is not None else ""

    return value


def get_int(key: str, default: int) -> int:
    """读取整数配置，非法值回落到默认值并保留原始文本便于排查。"""
    raw = get_setting(key, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def get_float(key: str, default: float) -> float:
    """读取浮点配置，非法值回落到默认值。"""
    raw = get_setting(key, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def get_bool(key: str, default: bool = False) -> bool:
    """读取布尔配置，接受 true/1/yes/on 四种写法（大小写不敏感）。"""
    raw = get_setting(key, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")