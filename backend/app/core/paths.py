"""路径工具：所有落盘位置都从 settings 派生，不出现写死的盘符。"""

from __future__ import annotations

from app.core.settings import PARSED_DIR, UPLOAD_DIR


def get_local_base_dir() -> str:
    """上传文件的本地归档根目录（按 日期/task_id 分子目录）。"""
    return str(UPLOAD_DIR)


def get_parsed_dir() -> str:
    """解析中间产物根目录（MinerU 输出、chunks.json 备份等）。"""
    return str(PARSED_DIR)