import os, logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


cache_llm_client = {}


def get_llm_client(mode_name: str = None, temperature: float = 0.0, response_format: bool = False):
    """
    Returns: 返回LLM客户端对象
    """

    # 1. 获取模型的名字,api_key和base_url
    model_name = mode_name or os.getenv('ITEM_MODEL', "qwen-flash")
    api_key = os.getenv('OPENAI_API_KEY')
    api_base = os.getenv('OPENAI_API_BASE')

    #2. get_llm_client方法中只有传入的参数和cache_key中完全一样，才不用创建新的模型实例（复用模型client）
    cache_key = (mode_name, response_format)  # 复合缓存key(a,b)
    # 缓存命中 直接返回
    if cache_key in cache_llm_client:
        return cache_llm_client[cache_key]



    # 3. 返回的内容格式
    model_kwargs = {}
    if response_format:     #如果要求返回的是json格式
        model_kwargs['response_format'] = {"type": "json_object"}
    try:
        # 4. 定义模型实例
        client = ChatOpenAI(
            model_name=model_name,
            openai_api_key=api_key,
            openai_api_base=api_base,
            temperature=temperature,
            extra_body={"enable_thinking": False},  #禁用思考
            model_kwargs=model_kwargs               #返回json格式
        )   #创建client成功

        # 5. 同步数据（将client添加到cache_llm_client中，下次使用可以直接调用，不用再创建）
        cache_llm_client[cache_key] = client

        # 6. 返回
        return client

    except Exception as e:
        logger.error(f"LLM客户端创建失败,原因:{str(e)}")
        return None
