from loguru import logger
from typing import List, Dict, Any
from pymilvus import MilvusClient, DataType
from config.settings import settings


class MilvusDB :
    """Milvus vector database for books and reviews
    """

    def __init__(self, collection_name: str = None,
                 vector_dim: int = None,
                 batch_size: int = None,
                  db_file: int = None):
        
        self.collection_name = collection_name or settings.collection_name

        self.vector_dim = vector_dim or settings.vector_dim

        self.batch_size = batch_size or settings.batch_size

        self.db_file = db_file or settings.milvus_db_path

        self.client = None 