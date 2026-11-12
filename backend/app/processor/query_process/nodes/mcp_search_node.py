import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union
from agents.mcp import MCPServerStreamableHttp  #MCPServerSse  # pip install openai_agents
from langchain_core.messages import SystemMessage, HumanMessage
from app.processor.query_process.state import QueryGraphState
from app.processor.query_process.base import BaseNode, T
from app.processor.query_process.exceptions import StateFieldError


class McpSearchNode(BaseNode):
    name = "mcp_search_node"
    """
     负责从网络查询当前的问题【整个知识库没有找到该问题，兜底的网络结果】
     mcp形式调用第三方的各种通用的搜索工具。
     百度：通用搜索工具、网页搜索、地图等（原项目为电商比价场景，此处仅保留通用搜索）
     灵积服务平台:的通用搜索工具【bailian_web_search】
     mcp: 本质：就是各大平台把通用的功能，封装成了工具（函数） 然后通过mcp协议 客户端就可以直接调用它。【mcp客户端】---->【mcp服务端：任意选择某一个】
     
    """

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:

        # 1. 未配置 MCP 端点时直接降级：联网检索是可选的兜底通道，
        #    不应该因为它没配就让整条问答链失败。
        if not self.config.mcp_dashscope_base_url:
            self.logger.info("未配置 MCP_DASHSCOPE_BASE_URL，跳过联网检索")
            return {"web_search_docs": []}

        # 2. 参数校验
        validated_rewritten_query, validated_item_names = self._validate_query_inputs(state)

        # 3. 创建mcp_client 并且让客户端执行工具  bailian_web_search
        mcp_result = asyncio.run(self._create_execute_web_search(validated_rewritten_query))

        # 3. 兜底：无论成功失败都保证 web_search_docs 这个字段存在（哪怕是空列表），
        #    否则下游（如 rerank_node）拿到 None 会直接报 TypeError
        if not mcp_result:
            self.logger.warning("MCP 搜索没有返回任何结果，web_search_docs 置为空列表")
            mcp_result = []

        # 4. 只更新state的web_search_docs
        return {"web_search_docs": mcp_result}

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:

        # 1. 获取state的rewritten_query
        rewritten_query = state.get('rewritten_query', "")

        # 2. 获取state的item_names
        item_names = state.get('item_names', "")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 4. 返回
        return rewritten_query, item_names

    async def _create_execute_web_search(self, validated_rewritten_query: str) -> List[Dict[str, Any]]:
        """
        1. 创建mcp客户端
        2. 客户端执行对mcp服务端调用
        别人的接口：1. 发送请求调用别人
        使用客户端要调用服务端【1. 原生发送请求 2. mcp客户端发送请求 3. Agent【mcp客户端发送请求】】
        Agent:代理【代理对象就是程序员】
        Args:
            validated_rewritten_query:
        Returns:
        """

        # 1. 创建mcp客户端【1.host 2. http 2.1 streamable(主要建议选择使用) http 2.2 sse http 3.stdio sse】
        mcp_client = MCPServerStreamableHttp(
            name="通用搜索",
            params={
                "url": self.config.mcp_dashscope_base_url,  # 服务端的端点
                "headers": {"Authorization": self.config.openai_api_key}  # 认证权限 api_key
            },
            cache_tools_list=True,  # mcp服务端工具列表的工具做缓存
        )
        try:
            # 2. 建立mcp连接
            await mcp_client.connect()

            # 3. 执行工具
            #    注意：search_pro 的入参 schema 是 {"query": str} 且 additionalProperties=false，
            #    多传 count 之类的参数会被服务端拒绝（PARAM_INVALID: Only param:query is supported）
            execute_tool_result = await mcp_client.call_tool(tool_name="search_pro",
                                                             arguments={"query": validated_rewritten_query})

            # 4. 解析工具执行完的结果
            # 4.1 获取最外层的对象
            if not execute_tool_result or not execute_tool_result.content:
                self.logger.error("MCP 工具调用返回为空")
                return []

            # 4.2 服务端返回错误时 isError=True，text 里是错误信息（不是搜索结果），必须先判断
            if getattr(execute_tool_result, "isError", False):
                self.logger.error(f"MCP 工具调用失败: {execute_tool_result.content[0].text}")
                return []

            # 4.3  获取TextContent对象的text
            text_content_text: str = execute_tool_result.content[0].text
            if not text_content_text:
                self.logger.error("MCP 工具调用返回的 text 为空")
                return []

            # 4.4 反序列化
            try:
                text_content_text: Dict[str, Any] = json.loads(text_content_text)

                # a) 获取pages
                pages = text_content_text.get('pages') or []
                if not isinstance(pages, list):
                    self.logger.error(f"MCP 结果中 pages 类型异常: {type(pages).__name__}")
                    return []
                if not pages:
                    self.logger.warning("MCP 结果的 pages 为空")
                    return []
                search_result = []
                # b) 遍历得到每一个结果
                for page in pages:
                    snippet = (page.get('snippet') or "").strip()
                    title = (page.get('title') or "").strip()
                    url = (page.get('url') or "").strip()
                    search_result.append({"snippet": snippet, "title": title, "url": url})
                # c) 最终返回
                return search_result
            except Exception as e:
                self.logger.error(f"反序列MCP结果失败: {e}")
                return []
        finally:
            await  mcp_client.cleanup()  # 关闭连接
