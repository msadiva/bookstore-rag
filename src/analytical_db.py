import duckdb
import pandas as pd
from typing import Dict, Any, List
from loguru import logger
from config.settings import settings


class AnalyticalDB:
    """DuckDB database for analytical queries on books data."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.analytical_db_path
        self.conn = duckdb.connect(self.db_path)
        logger.info(f"Connected to DuckDB at {self.db_path}")
    
    def create_books_table(self, df: pd.DataFrame):
        """Create books table from DataFrame with proper schema."""
        try:
            # Drop table if exists
            self.conn.execute("DROP TABLE IF EXISTS books")
            
            # Create table from DataFrame
            self.conn.execute("CREATE TABLE books AS SELECT * FROM df")
            
            # Get row count
            result = self.conn.execute("SELECT COUNT(*) FROM books").fetchone()
            row_count = result[0] if result else 0
            
            logger.info(f"Created books table with {row_count} rows")
            
        except Exception as e:
            logger.error(f"Error creating books table: {e}")
            raise
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get table schema information for SQL generation."""
        try:
            # Get column information
            result = self.conn.execute("DESCRIBE books").fetchall()
            
            schema = {
                "table_name": "books",
                "columns": {}
            }
            
            for row in result:
                column_name = row[0]
                column_type = row[1]
                schema["columns"][column_name] = column_type
            
            logger.info(f"Retrieved schema for books table: {len(schema['columns'])} columns")
            return schema
            
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return {"table_name": "books", "columns": {}}
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        try:
            logger.info(f"Executing SQL: {sql}")
            result = self.conn.execute(sql).fetchdf()
            logger.info(f"Query returned {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def validate_query(self, sql: str) -> bool:
        """Validate SQL query without executing it."""
        try:
            # Use EXPLAIN to validate without executing
            self.conn.execute(f"EXPLAIN {sql}")
            return True
        except Exception as e:
            logger.warning(f"SQL validation failed: {e}")
            return False
    
    def get_sample_data(self, limit: int = 5) -> pd.DataFrame:
        """Get sample data from books table."""
        try:
            return self.conn.execute(f"SELECT * FROM books LIMIT {limit}").fetchdf()
        except Exception as e:
            logger.error(f"Error getting sample data: {e}")
            return pd.DataFrame()
    
    def get_table_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the books table."""
        try:
            stats = {}
            
            # Total rows
            total_rows = self.conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
            stats["total_books"] = total_rows
            
            # Unique authors
            unique_authors = self.conn.execute("SELECT COUNT(DISTINCT author) FROM books").fetchone()[0]
            stats["unique_authors"] = unique_authors
            
            # Categories
            categories = self.conn.execute("SELECT DISTINCT book_category FROM books").fetchall()
            stats["categories"] = [row[0] for row in categories if row[0]]
            
            # Year range
            year_range = self.conn.execute("SELECT MIN(publication_year), MAX(publication_year) FROM books WHERE publication_year > 0").fetchone()
            if year_range and year_range[0] and year_range[1]:
                stats["year_range"] = {"min": year_range[0], "max": year_range[1]}
            
            # Rating range
            rating_range = self.conn.execute("SELECT MIN(average_rating), MAX(average_rating) FROM books WHERE average_rating > 0").fetchone()
            if rating_range and rating_range[0] and rating_range[1]:
                stats["rating_range"] = {"min": rating_range[0], "max": rating_range[1]}
            
            logger.info(f"Generated table stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Closed DuckDB connection")