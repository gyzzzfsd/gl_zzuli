import json

from langgraph.graph import StateGraph, END


from app.processor.import_process.nodes.entry_node import EntryNode
from app.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from app.processor.import_process.nodes.md_img_node import MarkDownImageNode
from app.processor.import_process.nodes.document_split_node import DocumentSplitNode
from app.processor.import_process.nodes.item_name_recognition_node import ItemNameRecognitionNode
from app.processor.import_process.nodes.bge_embedding_chunks_node import BgeEmbeddingChunksNode
from app.processor.import_process.nodes.import_milvus_node import ImportMilvusNode
from app.processor.import_process.nodes.kg_graph_node import KnowLedgeGraphNode
from app.processor.import_process.state import create_default_state,ImportGraphState
from app.processor.import_process.base import setup_logging



def import_router(state: ImportGraphState):
    if state.get('is_md_read_enabled'):
        return "md_img_node"
    if state.get('is_pdf_read_enabled'):
        return "pdf_to_md_node"
    return END  # 安全降级



def create_import_graph() -> StateGraph:
    """
    定义导入业务的graph状态拓扑谱（langgraph构建流水线）整个流水线各个节点要读取或者写入的节点。
    Returns:

    """

    # 1. 定义状态图
    graph_pineline = StateGraph(ImportGraphState)  # type:ignore

    # 2. 定义节点（入口、结束节点、自己需要添加的）
    # 2.1 定义入口节点
    graph_pineline.set_entry_point("entry_node")

    # 2.2 添加剩下的节点
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        'md_img_node': MarkDownImageNode(),
        'document_split_node': DocumentSplitNode(),
        'item_name_recognition_node':ItemNameRecognitionNode(),
        'bge_embedding_chunks_node': BgeEmbeddingChunksNode(),
        'import_milvus_node': ImportMilvusNode(),
        'knowledge_graph_node': KnowLedgeGraphNode(),

    }
    for key, value in nodes.items():
        graph_pineline.add_node(key, value)

    # 3. 定义边（顺序边、条件边）
    # TODO:条件边
    graph_pineline.add_conditional_edges("entry_node",
                                         import_router,
                                         {
                                             "md_img_node": "md_img_node",
                                             "pdf_to_md_node": "pdf_to_md_node",
                                             END: END
                                         }
                                         )


    #graph_pineline.add_edge("entry_node", "pdf_to_md_node")
    graph_pineline.add_edge("pdf_to_md_node", "md_img_node")
    graph_pineline.add_edge("md_img_node", 'document_split_node')
    graph_pineline.add_edge("document_split_node", 'item_name_recognition_node')
    graph_pineline.add_edge("item_name_recognition_node", 'bge_embedding_chunks_node')
    graph_pineline.add_edge("bge_embedding_chunks_node", 'import_milvus_node')
    graph_pineline.add_edge("import_milvus_node", 'knowledge_graph_node')
    graph_pineline.add_edge("knowledge_graph_node", END)



    # 4. 编译（编排）
    return graph_pineline.compile()




kb_import__graph_app = create_import_graph()





# 测试使用
def run_import_graph(import_file_path: str, file_dir: str):
    #接收参数 -> 组装初始数据 -> 启动流水线 -> 实时监控每一步的执行情况 -> 返回最终结果。
    """

    :param import_file_path: 要解析的文件路径
    :param file_dir: 解析后的文件目录
    :return:
    """
    # 1. 构建state:把传入的两个参数，装进一个普通的 Python 字典 state 中。
    state = {
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }
    #调用state.py中创建的create_default_state方法：这个方法是将原来默认state中的状态进行更新
    #只更新传入的**state中的内容，其余的仍然是默认值
    init_state = create_default_state(**state)  # 发生了解包



    # 2. 调用stream(用流式获取每一个节点的处理情况：event事件[节点名字 节点处理后的状态])
    final_state = None
    for event in kb_import__graph_app.stream(init_state):
        for node_name, state in event.items():
            print(f"运行节点的:{node_name},state:{state}")
            final_state = state #得到最后一个结点处理后的状态state

    return final_state
