from loguru import logger
from typing import List, Dict, Any
from pymilvus import MilvusClient, DataType, connections, FieldSchema, CollectionSchema, Collection
from config.settings import settings
from embed_data import EmbedData, batch_iterate


class MilvusDB :
    """Milvus vector database for books and reviews
    """

    def __init__(self, collection_name: str = None,
                 vector_dim: int = None,
                 batch_size: int = None,
                 host: str = None,
                 port: int = None ):
        
        self.collection_name = collection_name or settings.collection_name

        self.vector_dim = vector_dim or settings.vector_dim

        self.batch_size = batch_size or settings.batch_size

        self.host = host or settings.host

        self.port = port or settings.port

        self.client = None 


    def initilize_client(self) :
        """Connect to Milvus Standalone"""

        connections.connect(alias = "default", host = self.host, port = self.port)

    def create_collection(self):
        """Create collection with metadata for books and reviews."""
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("book_id", DataType.VARCHAR, max_length=64),
            FieldSchema("type", DataType.VARCHAR, max_length=20),  # 'book' or 'review'
            FieldSchema("book_category", DataType.VARCHAR, max_length = 20), ## either romance or comic graphics
            FieldSchema("context", DataType.VARCHAR, max_length=65535),  # title+desc or review
            FieldSchema("vector", DataType.BINARY_VECTOR, dim=self.vector_dim),
            FieldSchema("title", DataType.VARCHAR, max_length=512),
            FieldSchema("author", DataType.VARCHAR, max_length=256),
            FieldSchema("description", DataType.VARCHAR, max_length=65535),
            FieldSchema("avg_rating", DataType.FLOAT),
            FieldSchema("num_pages", DataType.INT64),
            FieldSchema("pub_year", DataType.INT64),
        ]
        schema = CollectionSchema(fields, description="Books and Reviews Collection")

        # Drop existing collection if exists
        if Collection.exists(self.collection_name):
            Collection(self.collection_name).drop()
            logger.info(f"Dropped existing collection: {self.collection_name}")

        self.collection = Collection(name=self.collection_name, schema=schema)
        logger.info(f"Created collection '{self.collection_name}'")

        # Create index for fast search
        index_params = {
            "index_type": "BIN_FLAT",
            "metric_type": "HAMMING",
            "params": {}
        }
        self.collection.create_index("vector", index_params)
        self.collection.load()
        logger.info("Collection loaded and index created.")

    def ingest_data(self, embed_data: EmbedData, metadata: List[Dict[str, Any]]):
        """Ingest embeddings and metadata (books or reviews)."""
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call create_collection() first.")

        total_inserted = 0
        for batch_context, batch_vectors, batch_meta in zip(
            batch_iterate(embed_data.contexts, self.batch_size),
            batch_iterate(embed_data.embeddings, self.batch_size),
            batch_iterate(metadata, self.batch_size)
        ):
            data_batch = []
            for context, vector, meta in zip(batch_context, batch_vectors, batch_meta):
                row = {
                    "context": context,
                    "vector": vector,
                    **meta
                }
                data_batch.append(row)

            self.collection.insert(data_batch)
            total_inserted += len(data_batch)
            logger.info(f"Inserted batch: {len(data_batch)} rows")

        logger.info(f"Successfully ingested {total_inserted} rows")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ):
        """Search similar items with optional filters (type, author, rating, etc.)."""
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call create_collection() first.")

        search_results = self.collection.search(
            data=[query_vector],
            anns_field="binary_vector",
            param={"metric_type": "HAMMING", "params": {}},
            limit=top_k,
            output_fields=["book_id","type","title","author","description","avg_rating"],
            expr=self._build_filter_expression(filters)
        )
        formatted = []
        for result in search_results[0]:
            formatted.append({
                "book_id": result.entity.get("book_id"),
                "type": result.entity.get("type"),
                "title": result.entity.get("title"),
                "author": result.entity.get("author"),
                "description": result.entity.get("description"),
                "book_category": result.entity.get("book_category"),
                "avg_rating": result.entity.get("avg_rating"),
                "score": result.score
            })
        return formatted

    def _build_filter_expression(self, filters: Dict[str, Any]) -> str:
        """Convert dict filters into Milvus expression string."""
        if not filters:
            return ""
        clauses = []
        for k, v in filters.items():
            if isinstance(v, str):
                clauses.append(f'{k} == "{v}"')
            else:
                clauses.append(f"{k} == {v}")
        return " and ".join(clauses)

    def get_collection_info(self):
        if not self.collection:
            return {"exists": False}
        stats = self.collection.num_entities
        return {"exists": True, "row_count": stats}

    def close(self):
        connections.disconnect("default")
        logger.info("Disconnected from Milvus Standalone")