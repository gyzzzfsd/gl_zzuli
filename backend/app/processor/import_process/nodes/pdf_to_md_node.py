import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Tuple

from app.processor.import_process.base import BaseNode, setup_logging
from app.processor.import_process.state import ImportGraphState
from app.processor.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError


def resolve_mineru_executable() -> str:
    """定位 mineru-kit 可执行文件。

    直接用 `python run.py` 启动时，conda 环境的 Scripts 目录并不在 PATH 里，
    subprocess 查找 "mineru-kit" 会抛 FileNotFoundError。这里按两级顺序解析：
    PATH → 当前 Python 解释器同级目录（conda 的 exe 都在那儿）。

    Returns:
        可执行文件的完整路径。

    Raises:
        PdfConversionError: 两个位置都没找到。
    """
    # 1. 优先用 PATH 里的（用户已 activate 环境的情况）
    found = shutil.which("mineru-kit")
    if found:
        return found

    # 2. 退回到当前解释器所在目录（conda: <env>/mineru-kit.exe，Linux: <env>/bin/mineru-kit）
    py_dir = Path(sys.executable).parent
    for root in (py_dir, py_dir / "Scripts", py_dir / "bin"):
        for name in ("mineru-kit.exe", "mineru-kit"):
            candidate = root / name
            if candidate.exists():
                return str(candidate)

    raise PdfConversionError(
        message="未找到 mineru-kit 命令，请确认 MinerU 已安装在当前 Python 环境：pip install mineru",
        node_name="pdf_to_md_node",
    )


class PdfToMdNode(BaseNode):
    """
      pdf转换md节点
    """
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """

        Args:
            self:
            state:

        Returns:
        """
        # 1. 对参数校验：得到输入文件路径和输出文件目录
        import_file_path, file_dir_path = self._validate_state_inputs_path(state)

        # 2. 利用MinerU工具解析pdf成为zip:得到解析状态processed_code=0表示解析成功
        processed_code = self._execute_mineru(import_file_path, file_dir_path)
        if processed_code != 0:
            raise PdfConversionError("MinerU解析PDF失败", self.name)

        # 3. 解压MinerU产出的zip：得到解压后的目录
        unzip_dir_path = self._extract_zip_file(import_file_path, file_dir_path)

        # 4. 获取md的path(解压出来的zip中的md文件)
        md_path = self._get_md_paths(unzip_dir_path)

        # 5. 更新state 字典的md_path
        state['md_path'] = md_path

        # 6. 返回state
        return state



    def _validate_state_inputs_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """

        Args:
            state:  该节点接收到的状态

        Returns:

        """
        self.log_step("step1", "对状态的路径输入参数做校验")

        # 1. 获取输入pdf文件路径
        import_file_path = state.get('import_file_path', '')

        # 2. 获取解析后的输出目录
        file_dir = state.get('file_dir', '')

        # 3.校验输出的文件路径(非空判断)
        if not import_file_path:
            raise ValidationError("解析的文件不存在", self.name)

        # 4. 用Path标准化
        import_file_path_opj = Path(import_file_path)

        # 5. 校验是一个真实的路径
        if not import_file_path_opj.exists():
            raise FileProcessingError("解析的文件路径不存在", self.name)

        # 6.判断输出目录是否为空
        if not file_dir:
            # 默认目录做兜底
            file_dir = import_file_path_opj.parent

        # 7. 标准输出目录
        file_dir_path_obj = Path(file_dir)
        self.logger.info(f"上传文件的路径:{import_file_path}")
        self.logger.info(f"输出的目录:{file_dir}")

        # 8.返回 输入文件以及输出目录的标准path
        return import_file_path_opj, file_dir_path_obj

    def _execute_mineru(self, import_file_path: Path, file_dir_path: Path):
        """

        Args:
            import_file_path:  要解析的文件路径
            file_dir_path:     解析后的文件目录

        Returns:
         mineru -p <input_path> -o <output_path>  --source local
        """
        self.log_step("step2", "执行MinerU解析PDF")

        # 1. 构建命令行（先解析出 mineru-kit 的绝对路径，理由见 resolve_mineru_executable）
        cmd = [
            resolve_mineru_executable(),
            "parse",
            str(import_file_path),
            "-o",
            str(file_dir_path),
            "-f",
            "zip",
            "--tier",
            "standard"
        ]
        import time
        process_start_time = time.time()
        # 2. 执行命令行(子进程执行命令行) 自动读取到主进程的环境变量
        proc = subprocess.Popen(args=cmd,           #要执行的命令。
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                errors="replace",  # ？ 或者空心的菱形
                                text=True,  # 输出的内容是字符串 不是字节
                                encoding="utf-8",  # 用指定的中文字符集  进行编解码
                                bufsize=1  # 按行缓冲区  只要缓冲区一行满了就给我
                                )

        #  3. 获取日志信息
        for line in proc.stdout:
            self.logger.info(f"执行MinerU产生的日志：{line}")

        #  4.等待子进程做完(主进程等待子进程做完) 如果主进程等到子进程做完得到的状态码是0 代表子进程成功执行完了。 反之，没有执行成功
        processed_code = proc.wait()

        process_end_time = time.time()
        if processed_code == 0:
            self.logger.info(
                f"MinerU成功解析PDF文件：{import_file_path.name} 耗时:{process_end_time - process_start_time:.2f}s")
        else:
            self.logger.error(f"MinerU解析PDF文件：{import_file_path.name}失败")

        # 5. 返回状态码
        return processed_code

    def _extract_zip_file(self, import_file_path: Path, file_dir_path: Path):
        """

        Args:
            import_file_path:  要解析的文件路径
            file_dir_path:     解析后的文件目录

        Returns:
            解压后的目录路径
        """
        self.log_step("step3", "解压MinerU产出的zip")
        # name: 名字.xxx  stem:只拿名字  后缀：suffix
        file_name = import_file_path.stem

        # 1. 拼出MinerU产出的zip路径  file_dir_path:D:\...\import_process\import_temp_dir
        #    zip_path:D:\...\import_process\import_temp_dir\万用表RS-12的使用.zip
        zip_path = file_dir_path / f"{file_name}.zip"

        # 2. 校验zip是不是真的产出了
        if not zip_path.exists():
            raise PdfConversionError(f"MinerU产出的zip文件不存在：{zip_path}", self.name)

        # 3. 解压目录 = 输出目录/文件名字  D:\...\import_process\import_temp_dir\万用表RS-12的使用
        unzip_dir_path = file_dir_path / file_name

        # 4. 目录不存在就创建(存在就复用，解压会覆盖同名文件)
        unzip_dir_path.mkdir(parents=True, exist_ok=True)

        # 5. 解压zip(zip的根目录下就是 markdown.md、images、middle_json.json 这些文件)
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(unzip_dir_path)

        self.logger.info(f"MinerU产出的zip解压完成：{zip_path.name} -> {unzip_dir_path}")

        # 6. 返回解压后的目录
        return unzip_dir_path

    def _get_md_paths(self, unzip_dir_path: Path):
        """
        """
        """
        :param unzip_dir_path: zip解压后的目录
        :return: 解压出来的md文件路径
        """

        # unzip_dir_path:D:\...\import_process\import_temp_dir\万用表RS-12的使用
        # 1. zip里的md文件名固定是 markdown.md
        md_path = unzip_dir_path / "markdown.md"

        # 2. 兜底：MinerU换版本后md文件名不叫markdown.md时，去解压目录里找md文件
        if not md_path.exists():
            md_file_list = list(unzip_dir_path.rglob("*.md"))
            if not md_file_list:
                raise PdfConversionError(f"解压后没有找到md文件：{unzip_dir_path}", self.name)
            md_path = md_file_list[0]

        return str(md_path)


###########测试
