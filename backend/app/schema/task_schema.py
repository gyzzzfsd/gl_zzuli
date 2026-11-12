"""
任务相关 Schema 定义
"""

from typing import List
from pydantic import BaseModel, Field


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    status: str = Field(..., description="任务状态：processing / completed / failed")
    done_list: List[str] = Field(..., description="已完成节点列表")
    running_list: List[str] = Field(..., description="正在运行节点列表")
    # 失败时带上原因，否则前端只能显示一个没有信息的"失败"
    error: str = Field("", description="失败原因（仅 failed 时有值）")