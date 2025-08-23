from loguru import logger
from typing import List, Dict, Any
from pymilvus import MilvusClient, DataType, connections, FieldSchema, CollectionSchema, Collection, utility
from config.settings import settings
from .embed_data import EmbedData, batch_iterate


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


    def initialize_client(self) :
        """Connect to Milvus Standalone"""

        connections.connect(alias = "default", host = self.host, port = self.port)

        logger.info("Suucessfully connected to Milvus Vector DB")

    def create_collection(self):
        """Create collection with metadata for books and reviews."""
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("book_id", DataType.VARCHAR, max_length=64, nullable=True),
            FieldSchema("book_category", DataType.VARCHAR, max_length = 20, nullable=True, default_value = "romance"), ## either romance or comic graphics
            FieldSchema("context", DataType.VARCHAR, max_length=65535, nullable=True),  # title+desc or review
            FieldSchema("vector", DataType.BINARY_VECTOR, dim=self.vector_dim),
            FieldSchema("title", DataType.VARCHAR, max_length=512, nullable=True),
            FieldSchema("author", DataType.VARCHAR, max_length=256, nullable=True),
            FieldSchema("description", DataType.VARCHAR, max_length=65535, nullable=True),
            FieldSchema("average_rating", DataType.FLOAT, nullable=True),
            FieldSchema("num_pages", DataType.INT64, nullable=True),
            FieldSchema("publication_year", DataType.INT64, nullable=True),
            FieldSchema("publisher", DataType.VARCHAR, max_length = 64, nullable=True),
            FieldSchema("price", DataType.FLOAT, nullable=True)
        ]
        schema = CollectionSchema(fields, description="Books and Reviews Collection")

        # Drop existing collection if exists
        # Drop existing collection if exists
        if utility.has_collection(self.collection_name):
            logger.info(f"Collection {self.collection_name} already exists")
            utility.drop_collection(self.collection_name)
            logger.info("Dropped the previous collection")
            self.collection = Collection(name=self.collection_name, schema=schema)
        else:
            # create collection
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
            batch_iterate(embed_data.binary_embeddings, self.batch_size),
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
        query_vector: bytes,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ):
        """Search similar items with optional filters (type, author, rating, etc.)."""
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call create_collection() first.")

        search_results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "HAMMING", "params": {}},
            limit=top_k,
            output_fields=["book_id","title","author","description","average_rating", "publication_year"],
            expr=self._build_filter_expression(filters)
        )
        formatted_results = []
        for result in search_results[0]:
            formatted_results.append({
            "id": result.id,
            "score": result.score,
            "payload": {
                "context": result.get("context") or "",
                "title": result.get("title") or "",
                "author": result.get("author") or "",
                "book_category": result.get("book_category") or "",
                "avg_rating": result.get("average_rating"),
                "pub_year": result.get("publication_year"),
            }
        })
        return formatted_results

    def _build_filter_expression(self, filters: Dict[str, Any]) -> str:
        if not filters:
            return ""

        expr_parts = []

        if filters.get("author"):
            expr_parts.append(f'author == "{filters["author"]}"')

        if filters.get("book_category"):
            expr_parts.append(f'book_category == "{filters["book_category"]}"')

        if filters.get("publication_year_min") is not None:
            expr_parts.append(f"publication_year >= {filters['publication_year_min']}")

        if filters.get("publication_year_max") is not None:
            expr_parts.append(f"publication_year <= {filters['publication_year_max']}")

        if filters.get("min_rating") is not None:
            expr_parts.append(f"average_rating >= {filters['min_rating']}")

        return " and ".join(expr_parts) if expr_parts else ""


    def get_collection_info(self):
        if not self.collection:
            return {"exists": False}
        stats = self.collection.num_entities
        return {"exists": True, "row_count": stats}

    def close(self):
        connections.disconnect("default")
        logger.info("Disconnected from Milvus Standalone")