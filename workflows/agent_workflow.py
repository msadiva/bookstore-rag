from typing import Optional, Any
from loguru import logger
from crewai import LLM
from crewai.flow.flow import Flow, start, listen, router, or_
from pydantic import BaseModel

from .events import RetrieveEvent, RouterOutput, QueryEvent, RagEvent
from src.retrieval import Retriever
from src.rag import RAG
from config.settings import settings

# Prompt templates for workflow steps
ROUTER_FILTER_PROMPT = """
You are a routing assistant that classifies user queries in a book search system. 
Your job is to (1) identify filters and (2) determine if the query is SEMANTIC or ANALYTICAL.

Definitions:
- SEMANTIC → The user wants to retrieve or explore books based on meaning, relevance, or similarity 
  (e.g., "Find me romance novels by Jane Austen", "Books similar to Pride and Prejudice").
- ANALYTICAL → The user wants to compute or analyze something using structured data 
  (e.g., "What is the average rating of romance books after 2010?", "How many books did J.K. Rowling publish after 2005?").

Possible book_category values: "romance", "comic", or null.

Respond ONLY in the following JSON format, filling null if not applicable: 

{schema}


USER QUERY:
{query}

JSON RESPONSE:
"""


# Define flow state
class BookStoreAgent(BaseModel):
    query: str = ""

class BookStoreAgentWorkflow(Flow[BookStoreAgent]):
    """Book Store Agent Workflow"""

    def __init__(
        self,
        retriever: Retriever,
        rag_system: RAG,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.rag = rag_system

        # Initialize OpenAI LLM for workflow operations
        self.llm = LLM(
            model=f"openai/{settings.llm_model}",
            temperature=0.1
        )
    def call_with_schema(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        """Utility to call LLM with enforced schema"""
        return self.llm.call(prompt=prompt, expected_output=schema)
    
    @start()
    def receive_query(self) -> QueryEvent:
        query = self.state.query
        if not query:
            raise ValueError("Query is required")
        return QueryEvent(query=query)

    

    @router(receive_query)
    def route_query(self, ev: QueryEvent) -> str:
        """Route Query"""

        routing_prompt = ROUTER_FILTER_PROMPT.format(
            query=ev.query,
            schema=RouterOutput.schema_json(indent=2)  # Pydantic schema JSON
        )

        routing_result = self.call_with_schema(routing_prompt, RouterOutput)

        logger.info(f"Routing decision: {routing_result.query_type}, Filters: {routing_result.filters}")
        
        self.state.filters = routing_result.filters        

        return "rag_flow" if routing_result.query_type.lower() == "semantic" else "non_semantic_flow"
    
    @listen("rag_flow")
    def run_rag_pipeline(self, ev: QueryEvent) -> RagEvent:
        """Execute RAG pipeline if query is semantic"""
        logger.info(f"Running RAG pipeline for query: {ev.query}")

        filters = getattr(self.state, "filters", None)
        
        rag_context = self.retriever.search(ev.query, filters=filters.dict() if filters else None)
        rag_answer = self.rag.query(ev.query, filters=filters.dict() if filters else None)

        return RagEvent(query=ev.query, rag_context=str(rag_context), answer=rag_answer)
    
    async def run_workflow(self, query: str) -> dict:
        """
        Run the semantic RAG workflow for a given query.
        """
        try:
            result = await self.kickoff_async(inputs={"query": query})
            return result if isinstance(result, dict) else {"answer": str(result), "query": query}
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "answer": f"Error while processing: {str(e)}",
                "rag_response": None,
                "query": query,
                "error": str(e)
            }
