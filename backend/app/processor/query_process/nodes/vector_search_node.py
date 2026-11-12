import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union
from app.processor.query_process.state import QueryGraphState
from app.processor.query_process.base import BaseNode, T
from app.processor.query_process.exceptions import StateFieldError
from app.utils.bge_m3_embedding_util import get_beg_m3_embedding_model,generate_hybrid_embeddings
from app.utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query


class VectorSearchNode(BaseNode):
    name = "vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState,Dict[str, Any]] :
        # 1. 参数校验
        #validated_query是item_name_confirm_node.py文件中传过来的rewritten_query
        #validate_item_names是item_name_confirm_node.py文件中传过来的confirmed
        ##confirmed中的内容形如 ["生物对象A", "生物对象B"]
        validated_query, validate_item_names = self._validate_query_inputs(state)

        # 2. 获取嵌入模型&milvus客户端
        embedding_model = get_beg_m3_embedding_model()
        milvus_client = get_milvus_client()
        # 降级只能返回本节点自己的 key：四路检索处于同一个并行超级步且没有 reducer，
        # 返回整个 state 会和其它分支写同一批键，触发 LangGraph 的 InvalidUpdateError。
        if embedding_model is None or milvus_client is None:
            self.logger.warning("嵌入模型或 Milvus 不可用，本路召回置空")
            return {"embedding_chunks": []}

        # 3. 对 validated_query 进行混合 向量化(稀疏向量做了字典的处理) 注意：【generate_hybrid_embeddings】
        embedding_result = generate_hybrid_embeddings(embedding_model,embedding_documents=[validated_query])
        if not embedding_result:
            self.logger.warning("查询向量化失败，本路召回置空")
            return {"embedding_chunks": []}

        # 4. 构建过滤表达式,item_name_filter_expr中的内容也形如 ["生物对象A", "生物对象B"]
        item_name_filter_expr = self._item_name_filter(validate_item_names)
        # 5. 创建混合搜索请求
        hybrid_requests = create_hybrid_search_requests(
            dense_vector=embedding_result['dense'][0],
            sparse_vector=embedding_result['sparse'][0],
            expr=item_name_filter_expr,
            limit=5
        )

        # 4. 执行混合搜索请求
        reps = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_requests,
            # ranker_weights=(0.5, 0.5),
            norm_score=True,#归一化，防止稀疏和稠密向量度量方式不同而造成的选择差异
            output_fields=["chunk_id", "content", "item_name"]
        )
        if not reps or not reps[0]:
            self.logger.warning("向量检索无命中，本路召回置空")
            return {"embedding_chunks": []}

        # 5. 更新state的embedding_chunks
        """
        把本路（稠密+稀疏混合检索）的原始命中列表整批塞进 state，供 RRF 融合使用。reps[0] 不是一个 chunk，而是"这一次查询的 top-5 命中列表"。
        reps                                    # List[List[dict]]   外层 = 查询条数 (validated_query的条数)
            └── reps[0]                            # List[dict]   ← 第0条查询（本节点只有这1条）的融合命中，长度 ≤ 5
                ├── {"chunk_id": 10241, "content":"测量电阻,将旋钮拨至 Ω 档……" , "item_name":  "RS-12 数字万用表"}   # 1条命中
                ├── {"chunk_id": 10087, "content": "测量前请确认表笔插孔位置……", "item_name":  "RS-12 数字万用表"}
                └── ...                                                   # 最多5条
        """
        #embedding_chunks是一个列表[{},{}......],形如：
        """
        [
            {
                "chunk_id": 464785677119629236,
                "distance": 0.7431622743606567,
                "entity": {
                    "chunk_id": 464785677119629236,
                    "content": "# 电阻测量\n\n\n警告：为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线......",
                    "item_name": "RS-12 数字万用表"
                }
            }            
            {
                "chunk_id": 464785677119629234,
                "distance": 0.7080745697021484,
                "entity": {
                    "chunk_id": 464785677119629234,
                    "content": "# 交流电压测量\n\n\n警告：谨防触电。\n\n若表笔长度不够不能接触到某些240V用具插座的带电部位......",
                    "item_name": "RS-12 数字万用表"
                }
            }
        ]
        """
        # 转成普通 dict 再入 state：pymilvus 返回的是 Hit(UserDict) 对象，
        # 不是 dict 子类，下游用 isinstance(x, dict) 判断会把它全部丢掉
        return {"embedding_chunks": [dict(h) for h in reps[0]]}

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

    def _item_name_filter(self, validate_item_names: List[str]) -> str:
        # filter = 'item_name in '"生物对象A", "生物对象B"'
        #  '"生物对象A", "生物对象B"'
        quoted = ", ".join(f'"{v}"' for v in validate_item_names)
        # filter = 'item_name in ["生物对象A", "生物对象B", "生物对象C"]'v   # 标量字段（动态字段）进行过滤
        return f" item_name in [{quoted}]"
