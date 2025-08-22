import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings) :

    llm_model_key: str = os.environ.get("LLM_Model_Key")

    llm_model: str = os.environ.get("LLM_Model")

    embedding_model: str = os.environ.get("Embedding_Model")

    vector_dim: int = os.environ.get("Embedding_Vector_Dim")

    host: str = os.environ.get("HOST")

    port: int = os.environ.get("PORT")

    top_k: int = 30
    batch_size: int = 512
    rerank_top_k: int = 10
    
    # Database Configuration
    milvus_db_path: str = "/data/milvus_binary.db"
    collection_name: str = "bookstore_agent"
    
    # Data Configuration
    docs_path: str = "data/"

    # Cache Configuration
    hf_cache_dir: str = "./cache/hf_cache"
    
    # LLM settings
    temperature: float = 0.3

def model_post_init(self, __context) -> None:
    # Create necessary directories
    Path(self.milvus_db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(self.hf_cache_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()