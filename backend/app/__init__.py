"""生物知识检索系统 —— 后端应用包。

这里先导入 settings，是为了让 `.env` 在**任何**业务模块读取环境变量之前完成加载。
Python 的导入顺序保证：`import app.xxx` 一定先执行本文件的语句。
"""

from app.core import settings  # noqa: F401  （导入即有副作用：加载 .env）

__all__ = ["settings"]