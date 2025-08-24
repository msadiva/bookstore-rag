from typing import Optional, Any, Dict
from loguru import logger
from crewai import LLM
from crewai.flow.flow import Flow, start, listen, router, or_
from pydantic import BaseModel

from .events import RetrieveEvent, RouterOutput, QueryEvent, RagEvent, AnalyticalEvent, SQLQuery
from src.retrieval import Retriever
from src.rag import RAG
from src.analytical_db import AnalyticalDB
from src.sql_generator import SQLGenerator
from config.settings import settings
import json

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
    filters: Optional[Dict[str, Any]] = None

class BookStoreAgentWorkflow(Flow[BookStoreAgent]):
    """Book Store Agent Workflow"""

    def __init__(
        self,
        retriever: Retriever,
        rag_system: RAG,
        analytical_db: AnalyticalDB = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.rag = rag_system
        self.analytical_db = analytical_db
        
        # Initialize SQL generator if analytical_db is provided
        if self.analytical_db:
            schema_info = self.analytical_db.get_schema_info()
            self.sql_generator = SQLGenerator(schema_info)
            logger.info("Initialized SQL generator with schema")

        # Initialize OpenAI LLM for workflow operations
        self.llm = LLM(
            model=f"openai/{settings.llm_model}",
            api_key=settings.llm_model_key,
            temperature=0.1
        )
    def call_with_schema(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        """Utility to call LLM with enforced schema"""
        self.llm.response_format = schema
        return self.llm.call(prompt)
    
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
            schema=json.dumps(RouterOutput.model_json_schema(), indent=2)
        )

        logger.info("Calling Router Agent")

        routing_result = self.call_with_schema(routing_prompt, RouterOutput)

        if isinstance(routing_result, str):
            routing_result = RouterOutput.model_validate(json.loads(routing_result))
        
        logger.info("Raw routing_result:", routing_result)
        logger.info(f"Routing decision: {routing_result.query_type}")
        
        self.state.filters = routing_result.filters        

        route_decision = "rag_flow" if routing_result.query_type.lower() == "semantic" else "non_semantic_flow"
        logger.info(f"Router returning: {route_decision}")
        
        # Debug analytical DB availability
        if route_decision == "non_semantic_flow":
            logger.info(f"Analytical DB available: {self.analytical_db is not None}")
            logger.info(f"SQL Generator available: {hasattr(self, 'sql_generator')}")
        
        return route_decision
    
    @listen("rag_flow")
    def run_rag_pipeline(self, ev: QueryEvent) -> RagEvent:
        """Execute RAG pipeline if query is semantic"""
        logger.info(f"Running RAG pipeline for query: {ev.query}")

        filters = getattr(self.state, "filters", None)
        
        rag_context = self.retriever.get_combined_context(ev.query, filters=filters.dict() if filters else None)
        rag_answer = self.rag.query(query=ev.query, context=rag_context)

        return RagEvent(query=ev.query, rag_context=str(rag_context), answer=rag_answer)
    
    @listen("non_semantic_flow")
    def run_analytical_pipeline(self, ev: QueryEvent) -> AnalyticalEvent:
        """Execute analytical pipeline for non-semantic queries"""
        print(f"🔥 ANALYTICAL PIPELINE STARTED FOR: {ev.query}")
        print(f"🔥 ANALYTICAL_DB EXISTS: {self.analytical_db is not None}")
        print(f"🔥 SQL_GENERATOR EXISTS: {hasattr(self, 'sql_generator')}")
        
        logger.info(f"Running analytical pipeline for query: {ev.query}")
        
        if not self.analytical_db or not hasattr(self, 'sql_generator'):
            # Fallback if no analytical DB is available
            print(f"🚨 ANALYTICAL PIPELINE FALLBACK - DB: {self.analytical_db is not None}, SQL_GEN: {hasattr(self, 'sql_generator')}")
            return AnalyticalEvent(
                query=ev.query,
                sql_query={
                    "sql": "",
                    "explanation": "Analytical database not available",
                    "confidence": 0.0
                },
                raw_results="",
                formatted_answer="Sorry, analytical queries are not available. Please try a semantic search instead."
            )
        
        try:
            # Step 1: Generate SQL query
            sql_query = self.sql_generator.generate_sql(ev.query)
            logger.info(f"Generated SQL: {sql_query.sql}")
            
            # Step 2: Validate SQL
            if not self.sql_generator.validate_generated_sql(sql_query):
                logger.warning("Generated SQL failed validation")
                return AnalyticalEvent(
                    query=ev.query,
                    sql_query={
                        "sql": sql_query.sql,
                        "explanation": sql_query.explanation,
                        "confidence": sql_query.confidence
                    },
                    raw_results="",
                    formatted_answer="I couldn't generate a valid SQL query for your request. Please try rephrasing your question."
                )
            
            # Step 3: Execute SQL query
            results_df = self.analytical_db.execute_query(sql_query.sql)
            raw_results = results_df.to_string(index=False)
            logger.info(f"Query returned {len(results_df)} rows")
            
            # Step 4: Format results with LLM
            formatted_answer = self._format_analytical_results(ev.query, sql_query, raw_results)
            
            analytical_event = AnalyticalEvent(
                query=ev.query,
                sql_query={
                    "sql": sql_query.sql,
                    "explanation": sql_query.explanation,
                    "confidence": sql_query.confidence
                },
                raw_results=raw_results,
                formatted_answer=formatted_answer
            )
            
            print(f"🔥 ANALYTICAL PIPELINE RETURNING: {analytical_event}")
            print(f"🔥 FORMATTED ANSWER: {formatted_answer}")
            
            return analytical_event
            
        except Exception as e:
            logger.error(f"Error in analytical pipeline: {e}")
            return AnalyticalEvent(
                query=ev.query,
                sql_query={
                    "sql": "",
                    "explanation": f"Error: {str(e)}",
                    "confidence": 0.0
                },
                raw_results="",
                formatted_answer=f"I encountered an error while processing your analytical query: {str(e)}"
            )
    
    def _format_analytical_results(self, query: str, sql_query: SQLQuery, raw_results: str) -> str:
        """Format analytical results using LLM."""
        format_prompt = f"""
You are a data analyst presenting query results to a user.

Original Question: {query}
SQL Query Used: {sql_query.sql}
Query Explanation: {sql_query.explanation}

Raw Results:
{raw_results}

Instructions:
1. Present the results in a clear, natural language format
2. Include key insights and numbers from the data
3. Make it easy to understand for non-technical users  
4. If results are empty, explain that no data matched the criteria
5. Keep the response concise but informative

Format this data into a helpful answer for the user.
"""
        
        try:
            print(f"🎨 FORMATTING PROMPT: {format_prompt}")
            
            # Create a fresh LLM instance to avoid context bleeding
            fresh_llm = LLM(
                model=f"openai/{settings.llm_model}",
                api_key=settings.llm_model_key,
                temperature=0.3
            )
            
            formatted_response = fresh_llm.call(format_prompt)
            print(f"🎨 LLM FORMATTED RESPONSE: {formatted_response}")
            return formatted_response
        except Exception as e:
            logger.error(f"Error formatting analytical results: {e}")
            print(f"🚨 FORMATTING ERROR: {e}")
            # Fallback formatting
            if raw_results.strip():
                fallback = f"Here are the results for your query '{query}':\n\n{raw_results}"
                print(f"🎨 USING FALLBACK: {fallback}")
                return fallback
            else:
                fallback = f"No data found matching your criteria: {query}"
                print(f"🎨 USING FALLBACK (NO DATA): {fallback}")
                return fallback
    
    async def run_workflow(self, query: str) -> dict:
        """
        Run the semantic RAG workflow for a given query.
        """
        try:
            result = await self.kickoff_async(inputs={"query": query})
            print(f"🔍 WORKFLOW RESULT TYPE: {type(result)}")
            print(f"🔍 WORKFLOW RESULT: {result}")
            
            # Handle different event types
            if hasattr(result, 'answer'):
                # RagEvent
                return {
                    "answer": result.answer,
                    "query": result.query if hasattr(result, 'query') else query,
                    "rag_context": result.rag_context if hasattr(result, 'rag_context') else None,
                    "event_type": "semantic"
                }
            elif hasattr(result, 'formatted_answer'):
                # AnalyticalEvent
                return {
                    "answer": result.formatted_answer,
                    "query": result.query if hasattr(result, 'query') else query,
                    "sql_query": result.sql_query.sql if hasattr(result.sql_query, 'sql') else None,
                    "event_type": "analytical"
                }
            elif isinstance(result, dict):
                # This might be the router output if flow didn't complete
                logger.warning(f"Received dict result instead of event: {result}")
                return {
                    "answer": f"Workflow returned unexpected result: {str(result)}",
                    "query": query,
                    "event_type": "error",
                    "debug_result": result
                }
            else:
                logger.warning(f"Unknown result type: {type(result)}, value: {result}")
                return {"answer": str(result), "query": query, "event_type": "unknown"}
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "answer": f"Error while processing: {str(e)}",
                "rag_response": None,
                "query": query,
                "error": str(e)
            }
