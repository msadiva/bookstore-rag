#!/usr/bin/env python3
"""
Test script for DuckDB analytical pipeline
Tests SQL generation and execution without the full Streamlit workflow
"""

import pandas as pd
from src.analytical_db import AnalyticalDB
from src.sql_generator import SQLGenerator
from config.settings import settings
import os

def test_analytical_pipeline():
    print("🧪 Testing DuckDB Analytical Pipeline\n")
    
    # Test 1: Load sample data
    print("1. Loading sample data...")
    try:
        # Try to load the preloaded CSV
        data_file = "data/final_vectorization_file.csv"
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            print("   Creating sample data instead...")
            
            # Create sample data
            sample_data = {
                'book_id': ['1', '2', '3', '4', '5'],
                'title': ['Book One', 'Book Two', 'Book Three', 'Book Four', 'Book Five'],
                'author': ['Author A', 'Author B', 'Author A', 'Author C', 'Author B'],
                'book_category': ['romance', 'comic', 'romance', 'romance', 'comic'],
                'publication_year': [2015, 2018, 2021, 2010, 2022],
                'average_rating': [4.2, 3.8, 4.7, 3.9, 4.1],
                'num_pages': [350, 200, 400, 280, 150],
                'price': [12.99, 8.99, 15.99, 11.99, 7.99],
                'publisher': ['Pub A', 'Pub B', 'Pub A', 'Pub C', 'Pub B'],
                'review_text': ['Great romance book', 'Funny comic', 'Amazing story', 'Good read', 'Nice comic']
            }
            df = pd.DataFrame(sample_data)
        else:
            df = pd.read_csv(data_file)
        
        print(f"✅ Loaded {len(df)} rows of data")
        print(f"   Columns: {list(df.columns)}")
        print()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Test 2: Initialize DuckDB
    print("2. Initializing DuckDB...")
    try:
        analytical_db = AnalyticalDB("test_books.duckdb")
        analytical_db.create_books_table(df)
        print("✅ DuckDB initialized and table created")
        print()
    except Exception as e:
        print(f"❌ Error initializing DuckDB: {e}")
        return

    # Test 3: Get schema info
    print("3. Testing schema retrieval...")
    try:
        schema_info = analytical_db.get_schema_info()
        print("✅ Schema retrieved:")
        for col, dtype in schema_info.get('columns', {}).items():
            print(f"   - {col}: {dtype}")
        print()
    except Exception as e:
        print(f"❌ Error getting schema: {e}")
        return

    # Test 4: Initialize SQL Generator
    print("4. Initializing SQL Generator...")
    try:
        sql_generator = SQLGenerator(schema_info)
        print("✅ SQL Generator initialized")
        print()
    except Exception as e:
        print(f"❌ Error initializing SQL Generator: {e}")
        return

    # Test 5: Test various queries
    test_queries = [
        "How many books are in the database?",
        "Count of romance books after 2010",
        "Show me books with rating above 4",
        "What is the average rating of romance books?",
        "How many books by each author?",
        "Books published after 2020"
    ]

    print("5. Testing SQL generation and execution...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Test {i}: {query}")
        try:
            # Generate SQL
            sql_result = sql_generator.generate_sql(query)
            print(f"   Generated SQL: {sql_result.sql}")
            print(f"   Confidence: {sql_result.confidence}")
            
            # Execute SQL
            if sql_result.confidence > 0.5:  # Only execute if confident
                results_df = analytical_db.execute_query(sql_result.sql)
                print(f"   Results: {len(results_df)} rows")
                if len(results_df) > 0:
                    print(f"   Sample result: {results_df.iloc[0].to_dict()}")
            else:
                print(f"   ⚠️ Skipping execution (low confidence)")
            
            print("   ✅ Test passed")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")

    # Test 6: Test table stats
    print("\n6. Testing table statistics...")
    try:
        stats = analytical_db.get_table_stats()
        print("✅ Table stats:")
        for key, value in stats.items():
            print(f"   - {key}: {value}")
        print()
    except Exception as e:
        print(f"❌ Error getting table stats: {e}")

    # Cleanup
    print("7. Cleaning up...")
    try:
        analytical_db.close()
        # Remove test database file
        if os.path.exists("test_books.duckdb"):
            os.remove("test_books.duckdb")
        print("✅ Cleanup completed")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

    print("\n🎉 Test completed!")

def test_simple_sql_generation():
    """Simple test for SQL generation only"""
    print("\n🔬 Testing SQL Generation Only\n")
    
    # Mock schema
    schema_info = {
        "table_name": "books",
        "columns": {
            "book_id": "VARCHAR",
            "title": "VARCHAR", 
            "author": "VARCHAR",
            "book_category": "VARCHAR",
            "publication_year": "INTEGER",
            "average_rating": "FLOAT",
            "price": "FLOAT"
        }
    }
    
    try:
        sql_gen = SQLGenerator(schema_info)
        
        test_queries = [
            "Count all books",
            "Romance books after 2010", 
            "Books with rating above 4.5",
            "Average price of books"
        ]
        
        for query in test_queries:
            print(f"Query: {query}")
            result = sql_gen.generate_sql(query)
            print(f"SQL: {result.sql}")
            print(f"Confidence: {result.confidence}")
            print()
            
    except Exception as e:
        print(f"❌ SQL Generation test failed: {e}")

if __name__ == "__main__":
    # Test SQL generation first (doesn't require data)
    # test_simple_sql_generation()
    
    # Test full pipeline
    test_analytical_pipeline()