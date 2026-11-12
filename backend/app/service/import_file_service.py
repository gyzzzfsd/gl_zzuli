from dotenv import load_dotenv

load_dotenv()
import os
import re
import os.path
import uuid
from datetime import datetime
import shutil
from pathlib import Path
from typing import Tuple
from fastapi import UploadFile, HTTPException
from app.core.paths import get_local_base_dir
from app.utils.minio_util import get_minio_client
from app.service.task_service import TaskService
from app.processor.import_process.state import ImportGraphState
from app.processor.import_process.main_graph import kb_import__graph_app
from app.utils.task_util import set_task_result

# 文件名里允许的字符：中文、字母、数字、下划线、点、括号、横线
_SAFE_NAME_RE = re.compile(r"[^\w\u4e00-\u9fa5.\-()（）]+")


def safe_filename(raw_name: str) -> str:
    """把客户端传来的文件名净化成一个安全的单层文件名。

    浏览器的 filename 是不可信输入：`../../etc/passwd` 这类值如果直接
    拼进 os.path.join，就能写到归档目录之外。这里只保留"名字"部分。

    Args:
        raw_name: 原始文件名（可能带路径）。

    Returns:
        安全文件名；原始名不可用时返回带时间戳的兜底名。
    """
    # 1. 去掉目录部分（Windows 和 POSIX 分隔符都要处理）
    name = os.path.basename(raw_name.replace("\\", "/")).strip()

    # 2. 过滤危险字符，并限制长度
    name = _SAFE_NAME_RE.sub("_", name).lstrip(".")[:120]

    # 3. 兜底：净化后什么都不剩（比如文件名全是特殊字符）
    if not name or name in (".", ".."):
        return f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}.dat"

    return name


class ImportFileService:
    """

    文件导入的业务类

    1. 保存上传文件到本地
    2. 保存上传文件到MinIO
    3. 运行图谱的所有节点
    """

    def __init__(self, task_service: TaskService):
        self._task_service = task_service

    def get_date_dir(self) -> str:
        return os.path.join(get_local_base_dir(), datetime.now().strftime('%Y%m%d'))

    def save_upload_file_to_local(self, file: UploadFile, file_dir: str):
        """
         将上传的文件保存到本地
        Args:
            file: 上传的文件
            file_dir: 文件的归档目录

        Returns:

        """

        # 1. 确保归档目录存在
        os.makedirs(file_dir, exist_ok=True)

        # 2. 构建上传文件的完整 path（文件名必须净化，防止 ../ 穿越）
        import_file_path = os.path.join(file_dir, safe_filename(file.filename or ""))

        # 3. 写入(批量的写入shutil.copyfileobj(file.file,f))
        with open(import_file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        # 4. 返回导入文件的path
        return import_file_path

    def save_upload_file_to_minio(self, import_file_path: str, file: UploadFile):
        """

        Args:
            import_file_path:
            file:

        Returns:

        """

        # 1. 获取minio客户端
        minio_client = get_minio_client()

        # 2. 判断minio客户端是否存在
        if not minio_client:
            raise HTTPException(status_code=500, detail="MinIO 服务不可用")

        # 3. 构建 MinIO 对象名：日期用 %Y%m%d（原实现写成 %Y%d%m，会把 10 月 9 日存成 0928），
        #    并带上 task 目录，避免同名文件互相覆盖
        date_part = datetime.now().strftime('%Y%m%d')
        task_part = Path(import_file_path).parent.name or "unknown"
        minio_object_name = f"origin_files/{date_part}/{task_part}/{safe_filename(file.filename or '')}"

        # 4. 获取桶名，并确保桶存在（首次部署时桶可能还没建）
        bucket_name = os.getenv("MINIO_BUCKET_NAME")
        try:
            if not minio_client.bucket_exists(bucket_name):
                minio_client.make_bucket(bucket_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MinIO 存储桶不可用: {e}")

        # 5. 开始上传
        try:
            minio_client.fput_object(bucket_name, minio_object_name, import_file_path)
        except Exception as e:
            raise ValueError(f"{file.filename}文件上传失败 原因:{e}")

    def process_upload_file(self, file: UploadFile) ->Tuple[str,str,str]:
        """
        处理上传文件
        Returns:
        1. 标记当前文件上传节点（"upload_file"）为正在运行中
        2. 将上传的文件保存到本地
        3. 将上传的文件保存到minio
        4. 标记当前文件上传节点（"upload_file"）为运行完毕
        5. 需要返回三部分信息（taski_id file_dir import_file_path）
        """

        # 1. 构建时间日期的文件目录出来
        date_dir = self.get_date_dir()

        # 2. 生成一个任务id
        task_id = str(uuid.uuid4())

        # 3. 构建文件的最终归属目录
        file_dir = os.path.join(date_dir, task_id)

        self._task_service.mark_node_running(task_id, "upload_file")
        # 4. 将接收到的文件上传本地
        import_file_path = self.save_upload_file_to_local(file, file_dir)

        # 5. 将本地磁盘的上传文件同步minio中
        self.save_upload_file_to_minio(import_file_path, file)
        self._task_service.mark_node_done(task_id, "upload_file")

        # 6. 构建返回值
        return task_id, file_dir, import_file_path

    def run_import_graph(self, task_id: str, file_dir: str, import_file_path: str):
        """执行导入流水线（在后台任务里跑）。

        Args:
            task_id: 任务 ID，前端用它轮询进度。
            file_dir: 本次任务的归档目录。
            import_file_path: 原始文件路径。

        Side effects:
            更新内存中的任务状态与节点进度；失败时把错误文本写进任务结果。
        """
        try:
            # 1. 标记任务开始处理
            self._task_service.update_task_status(task_id, "processing")

            # 2. 构建 LangGraph 初始状态
            global_graph_init_status: ImportGraphState = {
                "task_id": task_id,
                "file_dir": file_dir,
                "import_file_path": import_file_path
            }

            # 3. 流式执行：每拿到一个节点完成事件，就把它记进进度列表。
            #    原项目只在上传时标记过一次，前端进度条永远停在"上传文件"，
            #    这里补上逐节点的 done 记录。
            for event in kb_import__graph_app.stream(global_graph_init_status):
                for node_name in event.keys():
                    self._task_service.mark_node_done(task_id, node_name)

            # 4. 标记任务完成
            self._task_service.update_task_status(task_id, "completed")

        except Exception as e:
            # 5. 失败要落状态 + 落错误文本，否则前端只会看到一个转不完的圈
            self._task_service.update_task_status(task_id, "failed")
            set_task_result(task_id, "error", str(e))
            print(f"[{task_id}] Error: {e}")





