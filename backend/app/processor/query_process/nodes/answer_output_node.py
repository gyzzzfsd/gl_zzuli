"""答案输出节点：拼提示词 → 调 LLM → 清洗图片链接 → 落历史。

说明：模块名沿用原项目，实际职责是"生成并落地最终答案"。
"""

import re
from typing import List, Dict, Any, Tuple
from app.processor.query_process.base import BaseNode
from app.processor.query_process.state import QueryGraphState
from app.prompts.query.query_prompt import ANSWER_PROMPT
from app.utils.llm_client_util import get_llm_client
from app.utils.task_util import set_task_result
from app.utils.sse_util import push_sse_event, SSEEvent
from app.utils.mongo_history_util import save_chat_message


class AnswerOutputNode(BaseNode):
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        task_id = state.get("task_id")
        is_stream = state.get("is_stream")

        # 1. 已有答案 → 直接返回
        if state.get("answer"):
            self._push_existing_answer(state)

        # 2. 构建提示词 → 调用 LLM 生成答案
        else:
            prompt = self._build_prompt(state)
            state["prompt"] = prompt
            self._generate_answer(state, prompt)

        # 3. 写入历史记录（用户问题 + 助手回答）
        self._write_history(state)

        # 4. 流式模式发送结束事件
        if is_stream:
            push_sse_event(task_id, SSEEvent.FINAL,
                           {"answer": state.get("answer", "")})
        return state

    def _push_existing_answer(self, state: QueryGraphState):
        """非流式模式：存入任务结果；流式模式：让 FINAL 统一推送。"""
        if not state.get("is_stream"):
            set_task_result(state["task_id"], "answer", state["answer"])

    def _generate_answer(self, state, prompt):
        self.log_step("generate", "生成答案")
        llm_client = get_llm_client()
        if llm_client is None:
            raise ValueError("LLM 客户端初始化失败")

        task_id = state["task_id"]

        if state.get("is_stream"):
            state["answer"] = self._stream_generate(llm_client, prompt, task_id)
        else:
            state["answer"] = self._invoke_generate(prompt)

        # 清洗模型编造的图片链接：只保留检索结果里真实出现过的 URL
        state["answer"] = self._sanitize_image_block(state["answer"], state)

        # 非流式才需要写任务结果；流式由 FINAL 事件统一推送（推送的是清洗后的版本）
        if not state.get("is_stream"):
            set_task_result(task_id, "answer", state["answer"])

    def _sanitize_image_block(self, answer: str, state: QueryGraphState) -> str:
        """删除【图片】区块里不在检索结果中的 URL。

        模型经常无视 "只能用本地切片里的图片" 这条约束，凭空编出一个图片地址，
        前端会渲染成裂图。这里做一次白名单校验。

        Args:
            answer: 模型原始输出。
            state: 查询状态，用于取回检索到的文档。

        Returns:
            清洗后的答案；区块为空时整段删除。
        """
        if not answer or "【图片】" not in answer:
            return answer

        # 1. 收集允许出现的 URL（只来自检索到的切片正文与 url 字段）
        allowed: set[str] = set()
        for doc in (state.get("reranked_docs") or []) + (state.get("kg_chunks") or []):
            if not isinstance(doc, dict):
                continue
            for field in ("url", "content"):
                text = str(doc.get(field) or "")
                allowed.update(re.findall(r"https?://[^\s\)\]<>\"'，。]+", text))

        # 2. 拆出答案正文与图片区块
        head, sep, tail = answer.partition("【图片】")
        if not sep:
            return answer

        # 3. 逐行保留合法 URL，去重并保序
        kept: list[str] = []
        for line in tail.splitlines():
            url = line.strip().strip("<>").strip()
            if not url:
                continue
            if url in allowed and url not in kept:
                kept.append(url)

        # 4. 没有一张合法图片：把区块连同前面的分隔空行一起删掉
        if not kept:
            self.logger.info("模型给出的图片链接均不在检索结果中，已移除【图片】区块")
            return head.rstrip()

        return head.rstrip() + "\n\n【图片】\n" + "\n".join(f"<{u}>" for u in kept)

    def _build_prompt(self, state: QueryGraphState) -> str:
        char_budget = self.config.max_context_chars

        # 1. 获取问题和知识主体名
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state["item_names"]

        # 2. 格式化上下文文档
        context_str, char_budget = self._format_reranked_docs(
            state.get("reranked_docs") or [], char_budget
        )

        # 3. 格式化历史对话
        history_str, char_budget = self._format_chat_history(
            state.get("history") or [], char_budget
        )

        # 4. 格式化图谱关系
        graph_str, char_budget = self._format_kg_triples(
            state.get("kg_triples") or [], char_budget
        )

        # 5. 组装提示词
        return ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            item_names=", ".join(item_names),
            graph_relation_description=graph_str or "无图谱关系",
            question=question,
        )

    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue

            formatted_line = f"{role_label_map[role]}: {text}"
            used_chars += len(formatted_line) + 1

            if used_chars > char_budget:
                return "\n".join(formatted_lines), char_budget - used_chars

            formatted_lines.append(formatted_line)

        return "\n".join(formatted_lines), char_budget - used_chars

    def _format_reranked_docs(self, reranked_docs: List[Dict], char_budget: int) -> Tuple[str, int]:
        formatted_lines = []
        used_chars = 0

        for idx, doc in enumerate(reranked_docs, 1):
            content = doc.get("content", "").strip()
            if not content:
                continue

            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"), ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"), ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field, "")).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    @staticmethod
    def _format_kg_triples(kg_triples: List, char_budget: int) -> Tuple[str, int]:
        formatted_lines = []
        used_chars = 0
        for triple in kg_triples:
            triple_text = (str(triple) if triple is not None else "").strip()
            if not triple_text:
                continue
            if used_chars + len(triple_text) > char_budget:
                break
            formatted_lines.append(triple_text)
            used_chars += len(triple_text) + 1
        return "\n".join(formatted_lines), char_budget - used_chars

    def _invoke_generate(self, prompt: str) -> str:
        self.log_step("generate", "生成答案")
        llm_client = get_llm_client()
        if llm_client is None:
            raise ValueError("LLM 客户端初始化失败")
        try:
            response = llm_client.invoke(prompt)
            return response.content
        except Exception as e:
            self.logger.error(f"生成回答出错: {e}")
            return "抱歉，生成回答时出现错误。"

    def _stream_generate(self, llm_client, prompt, task_id):
        """流式生成，逐 chunk 推送 delta 事件。
         返回的是一个一个token(不是一个中文字符就是一个token )
        """
        accumulated_answer = ""
        try:
            for chunk in llm_client.stream(prompt):
                delta_text = getattr(chunk, "content", "") or ""
                if delta_text:
                    accumulated_answer += delta_text
                    push_sse_event(task_id, "delta", {"delta": delta_text})
        except Exception as e:
            self.logger.error(f"流式生成出错: {e}")
        return accumulated_answer

        # ★ 新增方法

    def _write_history(self, state: QueryGraphState):
        session_id = state["session_id"]
        rewritten_query = state.get("rewritten_query", "") or state.get("original_query", "")
        item_names = state.get("item_names") or []
        try:
            # 1. 写用户问题
            save_chat_message(
                session_id=session_id,
                role="user",
                text=state["original_query"],
                rewritten_query=rewritten_query,
                item_names=item_names,
            )
            # 2. AI回复（假的+真的）
            if state.get("answer"):
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=state["answer"],  # 模型的输出
                    rewritten_query=rewritten_query,
                    item_names=item_names,
                )
        except Exception as e:
            self.logger.warning(f"写入历史记录失败: {e}")
