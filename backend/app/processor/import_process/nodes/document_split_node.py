#############################################
#                    难！                    #
#                                           #
#############################################

import os, re, json
from typing import Tuple, List, Dict, Any
from app.processor.import_process.base import BaseNode, setup_logging
from app.processor.import_process.state import ImportGraphState
from app.processor.import_process.config import get_config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 加载--->打撒（1. 嵌入模型语义更准确 2.注入元数据 3.多路召回 4.性能、成本低----->减少LLM的幻觉，提高检索质量）---->组合:LLM就像人的脑子

        # 1. 获取参数:返回【md文件(按行读取)，md文件的文件名(不含扩展名)，切片最大长度，合并短内容的最小长度】
        md_content, file_title, max_content_length, min_content_length = self._get_inputs(state)

        # 2. 根据标题切割(核心)(将每个标题包括标题下的正文切分成一块)：  sections:[List[dict]
        """
        List[dict]是sections中的一个section
                   每一个section：
                                {
                                "title": "# 当前正在处理的章节的标题",
                                "body": "正文内容...",
                                "file_title": "md文件的文件名",
                                "parent_title": "# title的父标题
                                }
        """
        #sections是一个列表,里面装着被切分好的每一个小标题的内容 [{},{},{},{}.....]
        sections = self._split_by_headings(md_content, file_title)

        # 3. 处理(切分和合并)
        final_chunks = self.split_and_merge(sections, max_content_length, min_content_length)

        # 4. 组装
        chunks = self._assemble_chunk(final_chunks)

        # 5. 更新state:chunks
        state['chunks'] = chunks

        # 6. 日志统计
        self._log_summary(md_content, chunks, max_content_length)

        # 7. 备份
        state["chunks"] = chunks
        self._backup_chunks(state, chunks)

        # 8. 返回
        return state

        # ------------------------------------------------------------------ #
        #                       日志 & 备份                                    #
        # ------------------------------------------------------------------ #




    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):
        """输出切分统计信息"""
        self.log_step("step5", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        """将切分结果备份到 JSON 文件"""
        self.log_step("step6", "备份切片")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")

        except Exception as e:
            self.logger.warning(f"备份失败: {e}")

    def _get_inputs(self, state: ImportGraphState) -> Tuple[str, str, int, int]:

        self.log_step("step1", "切分文档的参数校验以及获取...")

        config = get_config()
        # 1. 获取md_content
        md_content = state.get('md_content')

        # 2. md_content文档统一换行符:把换行符统一成\n(因为windows有些换行符是\r\n,\r)
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3. 获取md_content文件标题（不含扩展名）
        file_title = state.get('file_title')

        # 4. 校验最大最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 or config.max_content_length <= config.min_content_length:
            raise ValueError(f"切片长度参数校验失败")
        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> List[dict]:
        """
    根据MD的标题（1-6）级标题进行切分:把一个完整的 Markdown 文本，按照 1到6级的标题，拆解成一个个带有上下文信息的“章节（section）”字典。
    Args:
        md_content: md内容
        file_title: 文档名字(不含扩展名)

    Returns:
    Tuple[List[dict], bool]
        List[dict]是sections中的一个section
                   sections中的每一个section：
                                        {
                                        "title": "# 第一章",
                                        "body": "正文内容...",
                                        "file_title": "万用表",
                                        "parent_title": "# 第一章"  父标题会更新
                                        }
        bool: md文档是否有标题
        """
        self.log_step("step2", "根据标题进行切分...")
        # 1. 定义变量
        in_fence = False     # 是否处于代码块内部（``` 或 ~~~）
        body_lines = []      # 当前章节的正文行（临时收集）
        sections = []        # 最终所有的章节列表
        current_level = 0    # 当前标题的级别（1-6）
        current_title = ""   # 当前标题的名字
        hierarchy = [""] * 7  # 7个长度 但是索引0不用 索引 1-6 分别存储当前生效的 1-6 级标题。




        # 2. 定义正则表达式(group1:标题的语法符号#【最少1个# 最多6个#】)
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")     #标题的正则表达式




        # 3. 切分:md_content按行切分
        content_lines = md_content.split("\n")

        def _flush():
            """
             封箱函数（核心）:封装section对象
            Returns:它的作用是：把当前积累的 body_lines 和 current_title 打包成一个 section 对象，存入 sections 列表
            """

            body = "\n".join(body_lines)    #body_lines 是一个字符串列表，每个元素是一行正文



            """
            如果文档开头就是一堆没有标题的正文，current_title = "" 但 body 有内容 → 需要封箱（用文件名做兜底标题）。
            如果文档里有空的标题（只有标题 没有正文），body = "" 但 current_title 有值 → 也需要封箱，记录这个标题。
            只有当两者都为空时（比如文档开头就是空的），才跳过封箱，避免生成无意义的空 section。
            """
            # 短路判断:只要“当前有标题” 或者 “当前有正文”，就值得封箱(封装成一个section)
            if current_title or body:
                parent_title = ""   #初始化父标题为空
                # 从当前等级标题的上一级开始查找，一直查找到一级标题（没有0级标题），为了找出父标题
                #一般来说：找当前标题的的上一级标题（父标题），直接往上找一级标题即可，根本不用写成循环的形式，一级一级往上找，
                # 这样做是为了防止md文档的标题不规范，比如当前标题是3级标题，而上一级标题是一级标题，没有二级标题，所以才要写成循环的形式，一级一级往上找父标题
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break

                if not parent_title:    #如果没有找到父标题（比如当前标题是一级标题，没有父标题，或者标题不规范，当前是二级标题，但是没有一级标题）
                    parent_title = current_title if current_title else file_title   #将父标题赋值为当前标题，如果没有那就赋值为md文档标题

                #返回的section形式：dict
                return sections.append({
                    "title": current_title if current_title else file_title,
                    "body": body,
                    "file_title": file_title,
                    "parent_title": parent_title
                })


        #判断代码块，防止将代码块中的注释当做标题
        for content_line in content_lines:
            # 3.1 判断是否存在代码块围栏
            if content_line.strip().startswith("```") or content_line.strip().startswith("~~~"):    #如果是代码块
                in_fence = not in_fence

            match = heading_re.match(content_line) if not in_fence else None  # None:假的值

            if match:
                _flush()

                level = len(match.group(1))  # 当前标题的级别
                current_level = level  # 当前标题的级别
                current_title = content_line
                hierarchy[level] = current_title  # 当前标题的名字

                for i in range(level + 1, 7):  # 清空
                    hierarchy[i] = ""

                body_lines = []
            else:
                # 除了标题行以外全都收集起来
                body_lines.append(content_line)

        _flush()

        return sections

    def split_and_merge(self, sections: List[Dict[str, Any]], max_content_length: int, min_content_length: int):
        """

        Args:
            sections: 根据 每个 标题切分后的所有section（章节）块
            max_content_length: 每一个section的content内容【title+body】长度最多不能超过指定 :将标题注入到内容中（标题注入：明确定位这一块的归属）
            min_content_length: 每一个section的content内容 长度如果比min_content_length小，尝试进行合并（合并：同源）

        Returns:
            List[section]

        """
        self.log_step("step3", "切分及合并...")

        # 1. 切分
        current_sections = []
        for section in sections:
            #注意最终current_sections的形式是： [{}, {}, ...]，因为这里用的是extend,而不是append
            # current_sections.extend( [] )的意思是，把[]中的元素逐个拆出来加进current_sections去
            current_sections.extend(self.split_long_section(section, max_content_length))

        # 2. 合并(再进行合并：合并的是两类，一类是body太小没有进split_long_section切分器直接return回current_sections的，看能否将其相邻的几个合并
        #                               第二类是切分器切分后的每个部分，看看能否合并（注意：切分器切分后自己就有合并逻辑，会将小的部分合并，所以这个再合并其实没有什么东西了）)
        final_sections = self.merge_short_section(current_sections, min_content_length)

        # 3. 返回
        return final_sections

    def split_long_section(self, section: Dict[str, Any], max_content_length: int):
        """
        只要满足条件的才会切（当前section的内容达到了max_content_length）(每一个section都要经过这个方法)
        Args:
            section:
            max_content_length:

        Returns:

        """

        self.log_step("step3", "进行长内容的切分")

        # 1. 获取section对象属性
        title = section.get('title')  # 不可能空
        body = section.get('body')  # 有可能是空
        file_title = section.get('file_title')  # 不可能是空
        parent_title = section.get('parent_title')  # 不可能是空

        # 2. 判断表格
        if "<table>" in body:
            self.logger.info("检测到了表格数据...")
            body = MarkdownTableLinearizer.process(body)

        # 3. 对标题做校验(感觉)
        TITLE_MAX_LENGTH = 50
        if len(title) > TITLE_MAX_LENGTH:
            self.logger.warning(f"检测文件{file_title}对应的{title}长度过长...")
            title = title[:50]

        # 4. 拼接title前缀
        title_prefix = f"{title}\n\n"

        # 5. 计算总长度(len(title_prefix)+len(body))
        total_length = len(title_prefix) + len(body)

        # 6. 判断小于或者刚好满足阈值（section内容比较短）
        if total_length <= max_content_length:
            return [section]

        # 7.计算body可用的长度
        body_length = max_content_length - len(title_prefix)

        if body_length <= 0:    #如果max_content_length太小，小于标题字符数
            return [section]    #不切了，原样返回

        # 8. 切分
        # 8.1 对谁切【body】 # 8.2 用谁切[1)手写 2)langchain提供的切分器:TextSpliter/TokenSpliter/LengthSpliter/）:chunkoverlap:块与块之间的重叠/:通用的递归切分器【"\n\n",'\n',' ',''[兜底：一个一个字符]】]

        # 8.1 定义递归的文档切分器对象
        #注意：能到这里的都是body_length > max_content_length - len(title_prefix) 的
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=body_length,
                                                       chunk_overlap=0,
                                                       separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";",
                                                                   " ", ""],
                                                       keep_separator=False
                                                       )
        # 8.2 进行切割
        """
        langchain提供的切分器TextSpliter的切分逻辑：
            第一步：切分
                把body传进去之后（body有很多段），直接先用\n\n去把body中的每一段都切开，
                然后检查切开的每一段，如果超过了body_length，那就使用\n将超过body_length的段落继续切分，
                然后这样使用separators中定义的切分逻辑重复执行，直到最后每一块的长度都不超过body_length。
                注意：切分逻辑不是按照separators定义的内容一遍遍去切分的，而是深度优先递归（如下所示：）
                不是"先全局按 \n\n 切完、再对超长段按 \n 切"。它是深度优先递归：碰到一个超长段落就当场钻进下一级分隔符处理它，处理完再回来继续。
            第二步：合并小段
                把切分后的小段拼起来，尽量填满max_content_length - len(title_prefix)的长度 。!!!这是切分器实现的功能
        """
        texts = text_splitter.split_text(body)     #这是只对body进行了切分，后面还要对切分出来的每一部分进行组装

        # 8.3 判断【长度：0 ：场景：body没有：长度：1:场景：标题下面的内容很少】
        if len(texts) <= 1:
            return [section]

        sub_section = []
        for index, text in enumerate(texts):
            sub_section.append({
                "title": title + "-" + f"{index + 1}",
                "body": text,
                "file_title": file_title,
                "parent_title": parent_title,
                "part": f"{index + 1}"  # 标记序号
            })

        # 8.4 返回
        return sub_section

    def merge_short_section(self, current_sections: List[Dict[str, Any]], min_content_length: int):
        """
        贪心累加算法:

        两个局限性：
        1. 撑破最小的阈值：大一点不用管
        2. 孤儿小块：也不用管（大量都是小块）
        Args:
            current_sections:
            min_content_length:
        Returns:

        """
        # 1. 定义变量
        current_section = current_sections[0]
        final_sections = []  # 最终的箱子

        # 2. 遍历以及合并
        for next_section in current_sections[1:]:
            # 同源
            same_parent = (current_section['parent_title'] == next_section['parent_title'])

            if same_parent and len(current_section.get('body')) < min_content_length:
                # body的合并(更新当前的section的body)
                current_section['body'] = (
                        current_section.get('body').rstrip() + "\n\n" + next_section.get('body').lstrip()
                )

                # 更新current_title  简单的能涵盖住合并进来的内容
                current_section['title'] = current_section['parent_title']
                current_section['part'] = 0
            else:
                #  将原来current_section进行封箱
                final_sections.append(current_section)
                #  更新next_section
                current_section = next_section

        # 最后一个人（封装起来）
        final_sections.append(current_section)

        # 4. 对所有section的part做处理 (为每一个父标题设置对应的part计数器)
        part_counter = {}
        result = []
        for final_section in final_sections:
            if "part" in final_section:
                # 获取section的父标题
                parent_title = final_section.get('parent_title')

                # 给计数器赋值
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1

                # 获取计数器的值
                new_part = part_counter[parent_title]

                final_section['part'] = new_part

                final_section['title'] = final_section['title'] + f"- {new_part}"

            result.append(final_section)

        return result

    def _assemble_chunk(self, final_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
         最终组合chunk
        Args:
            final_chunks:

        Returns:

        """

        self.log_step("step4", "组装最终的切片信息...")
        chunks = []
        for chunk in final_chunks:

            # 1. 获取chunk的信息
            title = chunk.get('title')
            file_title = chunk.get('file_title')
            parent_title = chunk.get('parent_title')
            body = chunk.get('body')
            content = f"{title}\n\n{body}"

            # 2. 构建最终chunk对象
            assemble_chunk = {
                "title": title,
                "file_title": file_title,
                "parent_title": parent_title,
                "content": content,
            }

            # 3.判断part是否存在
            if "part" in chunk:
                assemble_chunk['part'] = chunk.get('part')

            chunks.append(assemble_chunk)

        return chunks
