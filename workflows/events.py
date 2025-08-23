from typing import List, Optional, Literal
from pydantic import BaseModel
from src.retrieval import NodeWithScore

class FilterSchema(BaseModel):
    author: Optional[str] = None
    title: Optional[str] = None
    book_category: Optional[Literal["romance", "comic"]] = None  # restricted values
    publication_year_min: Optional[int] = None
    publication_year_max: Optional[int] = None
    min_rating: Optional[float] = None

class RouterOutput(BaseModel):
    query_type: Literal["semantic", "analytical"]
    filters: FilterSchema

class RetrieveEvent(BaseModel):
    """Event containing retrieved nodes from vector database."""
    retrieved_nodes: List[NodeWithScore]
    query: str

class QueryEvent(BaseModel):
    query: str

class RagEvent(BaseModel):
    query: str
    rag_context: str
    answer: str
