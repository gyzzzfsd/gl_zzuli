"""评估包的路径引导。

为什么单独一个模块：eval/ 在仓库根目录，业务代码在 backend/app 下。
不管从仓库根目录还是 backend/ 下执行脚本，都必须把 backend 加进 sys.path，
否则 `import app.*` 直接失败。集中在这里，避免每个脚本各写一遍。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"


def ensure_backend_on_path() -> Path:
    """把 backend 目录插入 sys.path 首位（幂等）。

    Returns:
        backend 目录的绝对路径。

    Raises:
        FileNotFoundError: backend/app 不存在，说明仓库结构被移动过。
    """
    # 1. 结构校验：宁可早失败，也不要等到 import 时报一个看不懂的 ModuleNotFoundError
    if not (BACKEND_DIR / "app").is_dir():
        raise FileNotFoundError(f"未找到业务代码目录：{BACKEND_DIR / 'app'}")

    # 2. 插入路径（放在最前面，避免装过同名 app 包的项目抢占）
    backend = str(BACKEND_DIR)
    if backend in sys.path:
        sys.path.remove(backend)
    sys.path.insert(0, backend)
    return BACKEND_DIR