from typing import Dict, Any
from loguru import logger
from crewai import LLM
from pydantic import BaseModel
from config.settings import settings
# Import will be added after fixing circular import
import json


class SQLGenerator:
    """Converts natural language queries to SQL using LLM."""
    
    def __init__(self, schema_info: Dict[str, Any]):
        self.schema_info = schema_info
        self.llm = LLM(
            model=f"openai/{settings.llm_model}",
            api_key=settings.llm_model_key,
            temperature=0.1  # Low temperature for consistent SQL generation
        )
        
        # SQL generation prompt template
        self.sql_prompt_template = """
You are an expert SQL analyst. Convert the natural language query to a valid DuckDB SQL query.

DATABASE SCHEMA:
Table: {table_name}
Columns and Types:
{column_info}

IMPORTANT RULES:
1. Use only the columns that exist in the schema above
2. Use proper DuckDB SQL syntax
3. For string comparisons, use single quotes
4. Available categories: 'romance', 'comic'
5. Be precise with column names (case-sensitive)
6. Use aggregate functions (COUNT, AVG, SUM, MIN, MAX) for analytical queries
7. Include proper WHERE clauses for filtering
8. Return only valid, executable SQL

EXAMPLE QUERIES:
- "How many books after 2020": SELECT COUNT(*) FROM books WHERE publication_year > 2020
- "Average rating of romance books": SELECT AVG(average_rating) FROM books WHERE book_category = 'romance'
- "Books by specific author": SELECT * FROM books WHERE author = 'Jane Austen'

USER QUERY: {user_query}

Respond with:
1. sql: The SQL query (string)
2. explanation: Brief explanation of what the query does (string)  
3. confidence: How confident you are this is correct (0.0 to 1.0)

{response_format}
"""
    
    def generate_sql(self, user_query: str):
        """Generate SQL query from natural language query."""
        # Import here to avoid circular import
        from workflows.events import SQLQuery
        
        try:
            # Format schema information
            column_info = "\n".join([
                f"- {col_name}: {col_type}" 
                for col_name, col_type in self.schema_info.get("columns", {}).items()
            ])
            
            # Create prompt using the template
            prompt = self.sql_prompt_template.format(
                table_name=self.schema_info.get("table_name", "books"),
                column_info=column_info,
                user_query=user_query,
                response_format="Return only the SQL query as plain text."
            )
            
            logger.info(f"Generating SQL for query: {user_query}")
            
            # Call LLM
            response = self.llm.call(prompt)
            print(f"Raw LLM response: {response}")
            
            # Clean the response to extract SQL
            sql_text = str(response).strip()
            # Remove any markdown formatting
            if '```' in sql_text:
                sql_text = sql_text.split('```')[1].replace('sql', '').strip()
            
            sql_query = SQLQuery(
                sql=sql_text,
                explanation=f"Generated SQL for: {user_query}",
                confidence=0.8
            )
            
            logger.info(f"Generated SQL: {sql_query.sql}")
            logger.info(f"Confidence: {sql_query.confidence}")
            
            return sql_query
            
        except Exception as e:
            print(f"Error = {e}")
            logger.error(f"Error generating SQL: {e}")
            # Return fallback SQL
            return SQLQuery(
                sql="SELECT COUNT(*) FROM books",
                explanation=f"Error generating SQL for: {user_query}. Showing total book count.",
                confidence=0.1
            )
    
    def _parse_fallback_response(self, response_text: str, user_query: str):
        """Fallback parser for non-JSON responses."""
        # Simple regex-based extraction
        import re
        
        # Try to extract SQL from response
        sql_match = re.search(r"SELECT.*?(?:;|$)", response_text, re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql = sql_match.group(0).replace(';', '').strip()
        else:
            sql = "SELECT COUNT(*) FROM books"
        
        from workflows.events import SQLQuery
        return SQLQuery(
            sql=sql,
            explanation=f"Extracted SQL query for: {user_query}",
            confidence=0.5
        )
    
    def validate_generated_sql(self, sql_query) -> bool:
        """Basic validation of generated SQL."""
        sql = sql_query.sql.strip().lower()
        
        # Basic checks
        if not sql.startswith('select'):
            return False
        
        if 'books' not in sql:
            return False
        
        # Check for dangerous operations
        dangerous_keywords = ['drop', 'delete', 'insert', 'update', 'alter', 'create']
        for keyword in dangerous_keywords:
            if keyword in sql:
                return False
        
        return True
    
    def get_sample_queries(self) -> Dict[str, str]:
        """Get sample queries for testing."""
        return {
            "count_all": "How many books are in the database?",
            "romance_count": "How many romance books are there?",
            "high_rated": "Show me books with rating above 4.5",
            "recent_books": "How many books were published after 2020?",
            "author_books": "How many books did each author write?",
            "avg_rating_by_category": "What's the average rating for each category?",
            "expensive_books": "Show me the most expensive books",
            "year_analysis": "How many books were published each year?"
        }