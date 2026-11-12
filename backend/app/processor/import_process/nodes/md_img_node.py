import time
import os, re
import base64
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, List, Deque
from openai import OpenAI
from app.utils.minio_util import get_minio_client
from app.processor.import_process.base import BaseNode, setup_logging
from app.processor.import_process.state import ImportGraphState
from app.processor.import_process.exceptions import ValidationError, FileProcessingError, ImageProcessingError
from app.processor.import_process.config import get_config


class MarkDownImageNode(BaseNode):
    """
    处理MarkDown图片节点类
    """
    name = "md_img_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """

        Args:
            state: 上一个节点处理之后state的最新状态

        Returns:   当前节点处理之后的state最新状态(md_content=process_md)

        """

        # 1. 获取配置对象
        config = get_config()



        # 2. 处理文件路径:返回md文件的全部内容，md文件的路径，图片所在的文件夹路径
        md_content, md_path_obj, image_dir = self._get_img_md_content(state)
        if not image_dir.exists():  #md文档没有图片需要处理
            # 图片不用处理了，直接更新state的md_content
            self.logger.warning(f"文件{md_path_obj.name}暂无图片要处理")
            state['md_content'] = md_content
            return state



        # 3. 扫描并处理图片（最复杂）: target_images_context列表包含了[图片名，图片路径 ，【图片所在的标题，图片的上文信息，图片的下文信息】 ]
        target_images_context = self._scan_images_and_context(image_dir, md_content, config)

        # 4. 用VLM给图片生成图片描述(摘要)
        images_summaries = self._extract_img_summary(md_path_obj.stem, target_images_context, config)

        # 5.复合函数
        # 5.1 本地图片上传到minio--->得到remote_url(图片远程的地址)
        # 5.2 替换md中的图片的本地url 以及vlm生成的摘要
        new_md_content = self._upload_img_and_update_new_md(md_path_obj.stem, md_content, images_summaries,
                                                            target_images_context, config)

        # 6. 将更新后的内容备份（调试）
        self._backup_new_md_file(md_path_obj, new_md_content)

        # 7. 更新state
        state['md_content'] = new_md_content

        # 8. 返回更新后的状态
        return state

    def _backup_new_md_file(
            self,
            md_path_obj: Path,
            new_md_content: str
    ) -> str:
        self.log_step("step_5", "备份新文件")

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )

        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(f"文件写入失败: {e}", node_name=self.name)

        return str(new_file_path)

    def _get_img_md_content(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """

        Args:
            state:  上一个节点处理之后state的最新状态

        Returns:
            md_content: md的内容
            md_path_obj: md的路径
            image_dir: 图片目录
        """
        self.log_step("step1", "读取md内容以及构建图片目录")
        # 1. 获取md_path
        md_path = state.get('md_path', '')

        # 2. 判断路径是否有内容
        if not md_path:
            raise ValidationError("md文件不存在", self.name)

        # 3. 标准化处理
        md_path_obj = Path(md_path)

        # 4. 判断路径是否有效
        if not md_path_obj.exists():
            raise FileProcessingError("md文件路径无效", self.name)

        # 5. 读取md内容
        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()  # 全部读取

        # 6.构建图片目录
        image_dir = md_path_obj.parent / "images"

        # 7. 返回:md文件的全部内容，md文件的路径，图片所在的文件夹路径
        return md_content, md_path_obj, image_dir

    def _scan_images_and_context(self, image_dir: Path, md_content: str, config) -> List[
        Tuple[str, str, Tuple[str, str, str]]]:

        """
        扫描图片并且 找到图片的上下文
        Args:
            image_dir: 图片目录
            md_content: md内容
            config：配置信息
        Returns:
          List[Tuple[str,str,Tuple[str,str,str]]]
          List[("图片名字",“图片的地址”,("离图片最近的上面一个标题","图片的上文","图片的下文"))]
        """

        self.log_step("step2", f"扫描图片文件目录{image_dir}")
        target_images_context = []

        # 1. 遍历图片文件目录
        for img_name in os.listdir(image_dir):  #img_name:得到每一个图片名
            # 1.1 获取文件的后缀
            file_ext = os.path.splitext(img_name)[1]  # a.txt

            # 1.2 判断后缀是否有效:判断img_name是否是一张图片，
            # 因为上面的for循环是遍历images文件夹下的所有文件，如果有文件不属于我们定义的图片类型，就直接丢弃
            if file_ext not in config.image_extensions:
                continue  # 继续处理下一个图片文件

            # 1.3 如果是图片：构建image_path(获取图片的路径) 转成字符串
            img_path = str(image_dir / img_name)

            # 1.4 构建图片（上下文）:返回了【图片所在的标题，图片的上文信息，图片的下文信息】
            img_context = self._find_img_context_with_limit(md_content, img_name, config.img_content_length)

            # 1.5 如果该图片没有上下文
            if not img_context:
                self.logger.warning("MD文件中暂未提取到可用的图片")
                continue  # 继续处理下一个图片文件

            # 1.6 提取到当前图片的唯一上下文内容(方便使用获取了第一个充当)
            primary_img_context = img_context[0]

            # 1.7 图片的完整信息构建到列表  :返回[图片名，图片路径 ，【图片所在的标题，图片的上文信息，图片的下文信息】 ]
            target_images_context.append((img_name, img_path, primary_img_context))

        # 1.8 返回所有图片完整信息
        self.logger.info(f"找到{len(target_images_context)}有效的图片")
        return target_images_context

    def _find_img_context_with_limit(self, md_content: str, img_name: str, img_content_length=200) -> List[Tuple[str, str, str]]:
        """
         从MD文档中提取图片上下文信息
         思路：使用正则查找图片在md的位置
        Args:
            md_content:  操作的MD
            img_name:    图片名字
            img_content_length:   最大的字符数限制
        Returns:
            List[Tuple[str, str, str]]
            List[("离图片最近的上面一个标题","图片的上文","图片的下文")]
        """
        # 1. 定义找图片的正则规则:  ![](img_name)
        re_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")

        # 2. 切分md内容
        md_lines = md_content.split("\n")   #将整个md文档按行切分
        imgs_context = []

        # 3.遍历md的每一行
        for line_idx, line in enumerate(md_lines):

            # 3.1 判断每一行line是否是图片
            if not re_pattern.search(line):
                continue  # 不是的话就继续下一行line


            #如果line是图片就执行以下内容
            # 3.2 找上文标题内容和标题索引
            head_title = ""  # 初始标题内容
            head_index = -1  # 初始的标题索引
            for i in range(line_idx - 1, -1, -1):   #从图片的上一行开始一行一行往上找，直到找到一个标题
                if re.match(r"^#{1,6}\s+", md_lines[i]):    #如果能匹配就是标题
                    head_title = md_lines[i]
                    head_index = i
                    break
            pre_content_start_index = head_index + 1    #图片上面的正文部分:从找到的标题下一行开始，一直到图片的上一行
            pre_content = md_lines[pre_content_start_index:line_idx]

            # 3.3 找上文的内容(自下而上,反转)
            img_pre_context = self._extract_img_context_with_limit(pre_content, img_content_length, direction="front")

            # 3.4 找下文标题索引(不要下文标题)
            section_index = len(md_lines)   #整个md文件的长度
            for i in range(line_idx + 1, section_index):    #从图片的下一行开始往下找，一直找到下一个标题或者文件结束
                if re.match(r"^#{1,6}\s+", md_lines[i]):
                    section_index = i
                    break

            post_content_start_index = line_idx + 1 #图片下面的正文部分:从图片的下一行开始找，一直找到下一个标题的上一行，或者文章结束
            post_content = md_lines[post_content_start_index:section_index]

            # 3.5 找下文的内容(正常顺序)
            img_post_context = self._extract_img_context_with_limit(post_content, img_content_length, direction="end")

            # 3.6 构建该图片的上下文信息: imgs_context列表包含了【图片所在的标题，图片的上文信息，图片的下文信息】
            imgs_context.append((head_title, img_pre_context, img_post_context))

        # 4.返回该md中当前图片的所有上下文信息（大多数情况下列表只要一个三元组对象） 除非该图片在md中有多处引用
        return imgs_context

    def _extract_img_context_with_limit(self, pre_content: list, img_content_length: int, direction: str) -> str:

        """
        提取图片到上下标题（最近）之间的内容（段落）
        direction：front:自下而上
        direction：end:自上而下
        如何从给定的内容中找段落？
        策略：md中的段落 \n分割  补充：行与行之间 每一行后面都有两个空格
        Args:
            pre_content:  提取到的内容
            img_content_length: 最大字符数
            direction: 放下昂

        Returns:
            str:上下文信息
        """
        current_paragraph = []  # 存储当前遍历到的内容
        final_paragraph = []  # 存储最终遍历到的段落（多个段落）

        # 1. 遍历每一行  收集段落
        for line in pre_content:
            clen_strip = line.strip()
            if not clen_strip:  # 自然而然的段落分割
                if current_paragraph:
                    final_paragraph.append("\n".join(current_paragraph))
                    current_paragraph = []
            else:
                if re.match(r"^!\[.*?\]\(.*?\)$", clen_strip):  # 遇到其它图片形成的段落
                    if current_paragraph:
                        final_paragraph.append("\n".join(current_paragraph))
                        current_paragraph = []
                    continue
                current_paragraph.append(line)

        # 2. 处理最后一个段落(最后一个段落后面没有段落分隔符，所以最后一个段落的内容只会放在current_paragraph中，所以要将其添加到final_paragraph中)
        if current_paragraph:
            final_paragraph.append("\n".join(current_paragraph))

        # 3.处理上文
        if direction == "front":    #如果是图片上面的内容，我们希望最终得到的是图片上面的一部分内容，而不是图片所在的标题下方的一部分内容
            final_paragraph.reverse()  # 找离图片最近的文档

        # 4. 收集最终返回的段落
        total = 0
        selected = []
        for para in final_paragraph:
            para_len = len(para)

            if total + para_len > img_content_length and selected:
                break
            selected.append(para)  # 放入
            total += para_len  # 跟新计数器

        # 5. 处理上文
        if direction == "front":
            selected.reverse()  # 返回的顺序和原文档的顺序是一致的

        # 6. 返回上下文
        return "\n\n".join(selected)

    def _extract_img_summary(self, document_title: str,
                             target_images_context: List[
                                 Tuple[str, str, Tuple[str, str, str]]],
                             config):
        """
         所有图片生成图片摘要 VLM (视觉语言模型)
        Args:
            document_title:  （文件）文档名字
            target_images_context: target_images_context列表包含了[图片名，图片路径 ，【图片所在的标题，图片的上文信息，图片的下文信息】 ]
            config: 配置信息

        Returns:  Dict{"图片名字1":"摘要","图片名字2":"摘要2"}

        """
        self.log_step("step3", "准备提取图片摘要")

        summaries = {}
        request_timestamps: Deque[float] = deque()
        # 1. 构建OpenAI客户端(超时+重试，避免单个请求把整个节点卡死)
        try:
            client = OpenAI(
                api_key=config.openai_api_key,
                base_url=config.openai_api_base,
                timeout=config.llm_timeout_seconds,  # 单次请求最长等多久
                max_retries=config.llm_max_retries   # 失败后重试几次
            )
        except Exception as e:
            self.logger.error(f"VLM客户端创建失败: {e}")
            return summaries

        # 2. 限流用的时间戳队列在多个线程之间共享，所以要用一把锁保护起来
        rate_limit_lock = threading.Lock()

        # 3. 并发发送请求: 图片之间互不依赖，可以同时发给VLM，不用一张一张排队等
        #   max_workers: 并发线程数；图片数量比线程数少时，就没必要开那么多线程
        max_workers = max(1, min(config.img_summary_max_workers, len(target_images_context)))
        self.logger.info(f"开启{max_workers}个线程，并发提取{len(target_images_context)}张图片的摘要")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            # 3.1 提交任务: {Future对象: 图片名}  后面拿到Future就能知道是哪个图片的结果
            future_to_img_name = {
                executor.submit(
                    self._get_img_summary,
                    config,
                    client,
                    document_title,
                    img_path,
                    images_context,
                    request_timestamps,
                    rate_limit_lock
                ): img_name
                for img_name, img_path, images_context in target_images_context
            }

            # 3.2 谁先跑完就先收谁的(主线程统一往summaries里写，避免多个线程同时写同一个字典)
            for future in as_completed(future_to_img_name):
                img_name = future_to_img_name[future]
                try:
                    summaries[img_name] = future.result()
                    self.logger.info(f"已完成{len(summaries)}/{len(future_to_img_name)}张图片摘要")
                except Exception as e:
                    # 单张图片失败不能影响其它图片
                    self.logger.warning(f"图片摘要任务异常 {img_name}: {e}")
                    summaries[img_name] = "图片描述"

        # 4. 返回映射表
        self.logger.info(f"生成{len(summaries)}图片摘要")

        return summaries

    def _enforce_rate_limit(
            self,
            request_timestamps: Deque[float],
            max_requests: int,
            rate_limit_lock: threading.Lock,
            window_seconds: int = 60
    ):
        """
        强制执行 API 请求速率限制。

        Args:
            request_timestamps (Deque[float]): 请求时间戳队列。
            max_requests (int): 窗口内最大请求数。
            rate_limit_lock (threading.Lock): 保护时间戳队列的线程锁。
            window_seconds (int, optional): 时间窗口大小（秒）。
        """
        while True:
            # 1. 加锁: 判断当前窗口还允不允许发请求，允许就把时间戳记进去、然后放行
            with rate_limit_lock:

                current_time = time.time()

                # 2. 移除窗口外的时间戳
                while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
                    request_timestamps.popleft()

                # 3. 没到上限: 记录时间戳后直接返回
                if len(request_timestamps) < max_requests:
                    request_timestamps.append(current_time)
                    return

                # 4. 到上限了: 算出还要等多久
                #    注意sleep必须放到锁外面，否则会把其它一起并发的线程全堵死
                sleep_duration = window_seconds - (current_time - request_timestamps[0])

            if sleep_duration > 0:
                self.logger.info(f"达到速率限制，暂停 {sleep_duration:.2f} 秒...")
                time.sleep(sleep_duration)

    def _get_image_mime(self, img_path: str) -> str:
        """
        按图片后缀推导MIME类型

        :param img_path: 图片路径
        :return: MIME类型; 认不出来就按jpeg兜底
        """
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }

        # 1. 取后缀(统一转小写，避免大写后缀匹配不上)
        file_ext = os.path.splitext(img_path)[1].lower()

        # 2. 返回MIME类型
        return mime_map.get(file_ext, "image/jpeg")

    def _get_img_summary(self, config, client, document_title: str, img_path: str,
                         images_context: Tuple[str, str, str],
                         request_timestamps: Deque[float], rate_limit_lock: threading.Lock) -> str:
        """

        :param config:
        :param client:
        :param document_title:  md文件的文件名
        :param img_path: 图片路径
        :param images_context: 图片的上下文信息
        :param request_timestamps: 请求时间戳队列(限流用)
        :param rate_limit_lock: 保护时间戳队列的线程锁
        :return:
        """

        # 0. 先过限流: 这个方法是多线程跑的，所以限流要放在这里，不能只放在提交任务的主循环里
        self._enforce_rate_limit(request_timestamps, config.requests_per_minute, rate_limit_lock, 60)

        # 1. 解包images_context构建上下文
        section_title, pre_context, post_contex = images_context

        # 2. 判断上下文
        context_parts = []
        if section_title:
            context_parts.append(section_title)
        if pre_context:
            context_parts.append(pre_context)
        if post_contex:
            context_parts.append(post_contex)

        # 3. 构建上下文
        final_context = "\n".join(context_parts) if context_parts else "暂无可用上下文"

        # 4. 读取图片文件
        local_img_content = ""
        try:
            with open(img_path, "rb") as f:
                local_img_content = base64.b64encode(f.read()).decode("utf-8")
            # 5. 发送请求
        except Exception as e:
            return "暂无图片"

        # 4.1 按后缀推导真实的MIME类型(不能硬编码image/jpeg，png/webp这些会把类型传错)
        image_mime = self._get_image_mime(img_path)

        # 5.调用 VLM
        try:
            response = client.chat.completions.create(
                model=config.vl_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""任务：为Markdown文档中的图片生成一个简短的中文标题。
                                背景信息：
                                    1. 所属文档标题："{document_title}"
                                    2. 图片上下文：{final_context}
                                    请结合图片视觉内容和上述上下文信息，用中文简要总结这张图片的内容，
                                    生成一个精准的中文标题（不要包含"图片"二字）。""",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_mime};base64,{local_img_content}"
                                }
                            }
                        ]
                    }
                ]
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败 {img_path}: {e}")
            return "图片描述"

    def _upload_img_and_update_new_md(self, document_name, md_content, images_summaries, target_images_context, config

                                      ):

        """
        上传图片到minio以及替换md中的图片url和摘要
        Args:
            md_content:
            images_summaries:
            target_images_context:

        Returns:

        """
        self.log_step("step5","上传图片到minio并且更新md的摘要和图片地址")

        remote_urls = {}
        # 1. 构建MinIO客户端
        minio_client = get_minio_client()

        if minio_client is None:
            self.logger.warning(f"无法将本地的图片上传到minio")  # 可选

        # 2. 遍历图片信息列表:target_images_context列表包含了[图片名，图片路径 ，【图片所在的标题，图片的上文信息，图片的下文信息】 ]
        for img_name, img_path, _ in target_images_context:

            # 2.1 构建对象的名字:document_name是md文件的名字，不包含后缀
            object_name = f"{document_name}/{img_name}"

            try:
                # 2.2 开始上传
                minio_client.fput_object(
                    config.minio_bucket,    #桶
                    object_name,            #图片名
                    img_path,               #图片路径（本地）
                )
                # 2.3 手动拼接远程地址
                # http://192.168.200.130:9000/test/temp_3.png
                remote_url = config.get_minio_base_url() + "/" + config.minio_bucket + "/" + object_name    #图片在minio桶中可以直接访问的地址（远程地址）
                self.logger.info(f"{img_name}图片上传到minio成功")
                remote_urls[img_name] = remote_url

            except Exception as e:
                self.logger.warning(f"{img_name}上传到minio失败")
                remote_urls[img_name] = "http://minio_mock/" + document_name + "/" + img_name  # 可选

        self.logger.info(f"成功上传{len(remote_urls)}图片到minio")

        # 3. 替换（摘要和图片地址）到MD内容中
        new_md_content = md_content
        for img_name, images_summary in images_summaries.items():
            # 3.1 提取远程地址
            remote_url = remote_urls.get(img_name)

            if not remote_url:
                continue  # 不替换md中该图片

            # 3.2 替换url和摘要
            replace_pattern = re.compile(
                r"!\[(.*?)\]\((.*?" + re.escape(img_name) + r".*?)\)",
                re.IGNORECASE
            )
            new_md_content = replace_pattern.sub(f"![{images_summary}]({remote_url})", new_md_content)

        return new_md_content
