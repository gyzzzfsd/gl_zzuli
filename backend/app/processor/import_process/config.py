"""
导入流程配置管理模块

集中管理所有配置项，支持环境变量覆盖
"""

from dataclasses import dataclass, field
from typing import Set, Optional
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# 本模块所在目录：app/processor/import_process
_IMPORT_PROCESS_DIR = Path(__file__).resolve().parent

@dataclass
class ImportConfig:
    """导入流程配置"""

    # ==================== 文档处理配置 ====================
    max_content_length: int = 2000  # 切片最大长度
    img_content_length: int = 200   #图片上下文长度
    min_content_length: int = 500   # 合并短内容的最小长度
    overlap_sentences: int = 1      # 句子级切分时的重叠句数
    item_name_chunk_k: int = 3      # 知识主体名识别时使用的切片数量
    item_name_chunk_size: int = 2500  # 知识主体名识别时使用的切片内容长度

    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    vl_model: str = field(
        default_factory=lambda: os.getenv("VL_MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )
    # 单次LLM请求的超时时间(秒): 超过就抛错重试, 避免一个请求把整个节点永久卡死
    llm_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("LLM_TIMEOUT_SECONDS", "300"))
    )
    # 失败后的重试次数(交给openai客户端自己重试)
    llm_max_retries: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "1"))
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )

    # ==================== Neo4j 配置 ====================
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "")
    )
    neo4j_username: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "")
    )
    neo4j_database: str = field(
        default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j")
    )

    # ==================== MinIO 配置 ====================
    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
    )
    minio_secure: bool = False

    # ==================== 向量配置 ====================
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )
    embedding_batch_size: int = 5

    # ==================== 速率限制 ====================
    requests_per_minute: int = 12  # 图片总结 API 速率限制
    # 图片摘要的并发线程数: 图片之间互不依赖, 可以同时发给VLM, 不用一张一张等
    img_summary_max_workers: int = field(
        default_factory=lambda: int(os.getenv("IMG_SUMMARY_MAX_WORKERS", "4"))
    )

    def get_minio_base_url(self):
        base_protocol = 'https://' if self.minio_secure else 'http://'
        return base_protocol + f"{self.minio_endpoint}"



    # ==================== PDF 转 Markdown（MinerU）配置 ====================
    # PDF 输入目录：留空表示由上传接口逐任务传入，不再写死某个盘符
    pdf_input_dir: str = field(
        default_factory=lambda: os.getenv("PDF_INPUT_DIR", "")
    )
    # Markdown 输出目录：同上，实际由 state["file_dir"] 决定
    md_output_dir: str = field(
        default_factory=lambda: os.getenv("MD_OUTPUT_DIR", "")
    )
    # 解析档位：flash / basic / standard / advanced；standard = 小模型 + VLM（混合模式）
    mineru_tier: str = field(
        default_factory=lambda: os.getenv("MINERU_TIER", "standard")
    )
    # MinerU 模型根目录（必须与下载时一致，否则会去 ~/.mineru/models 找模型）
    # 不再写死盘符：由 .env 的 MINERU_MODEL_BASE_DIR 决定，缺省则交给 MinerU 自己找
    mineru_model_base_dir: str = field(
        default_factory=lambda: os.getenv("MINERU_MODEL_BASE_DIR", "")
    )
    # 模型来源：modelscope / huggingface / auto
    mineru_model_source: str = field(
        default_factory=lambda: os.getenv("MINERU_MODEL_SOURCE", "modelscope")
    )
    # 本地 VLM 引擎：留空=自动；可选 lmdeploy / vllm / llama-cpp
    mineru_vlm_engine: str = field(
        default_factory=lambda: os.getenv("MINERU_MODEL_VLM_ENGINE", "")
    )
    # 找不到 mineru 包时，退回调用的 CLI 可执行文件（留空则自动在当前环境里找）
    mineru_kit_exe: str = field(
        default_factory=lambda: os.getenv("MINERU_KIT_EXE", "")
    )
    # 图片处理模式：extract=外置成文件并改写链接 | strip=直接丢弃 | inline=保留 base64
    pdf_images_mode: str = field(
        default_factory=lambda: os.getenv("PDF_IMAGES_MODE", "extract")
    )
    # strip 模式下图片被替换成的占位文本
    image_placeholder: str = "[图片]"
    # 单次解析最长等待秒数（仅为兜底，正常由 MinerU 自己控制）
    pdf_parse_timeout: int = field(
        default_factory=lambda: int(os.getenv("PDF_PARSE_TIMEOUT", "3600"))
    )

    @classmethod
    def from_env(cls) -> "ImportConfig":        #提供一个从环境变量读取配置的统一入口
        """从环境变量加载配置"""
        return cls()



# ==================== 全局单例 ====================
_config: Optional[ImportConfig] = None  #以_开头的变量名是约定只能在本模块内部用，将_config的初值设为None


def get_config() -> ImportConfig:       #
    """获取配置单例"""
    global _config
    if _config is None:     #懒加载:“如果全局配置还没被创建过（是 None），我才去创建它。” 如果已经创建过了，就跳过这一步(只用创建一次)
        _config = ImportConfig.from_env()   #调用工厂方法 from_env()，从环境变量加载配置，创建一个实例
    return _config

