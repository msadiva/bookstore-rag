# 📚 Bookstore RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system for book recommendations and queries, built with modern AI technologies and featuring both vector-based semantic search and analytical SQL capabilities.

![System Design](assetts/Rag%20on%20BookStore%20Dark.png)

## 🎯 Overview

This system provides an intelligent chatbot interface for querying and discovering books from a curated dataset of romance and comic/graphic novel categories. The system combines semantic search using vector embeddings with analytical SQL queries through DuckDB, offering users both contextual recommendations and precise data analysis.

## 🏗️ System Architecture

The system follows a modern microservices architecture with the following key components:

- **Vector Database (Milvus)**: Stores book embeddings for semantic similarity search
- **Analytical Database (DuckDB)**: Handles structured queries and aggregations
- **Embedding Service**: Uses BGE models for high-quality text embeddings
- **RAG Pipeline**: Combines retrieval with LLM generation for comprehensive responses
- **Agent Workflow**: Orchestrates complex multi-step queries and responses
- **Streamlit Frontend**: Provides an intuitive web interface

## 📊 Dataset

The system contains books from two main categories sourced from Goodreads:

### Romance Books
- **Source**: Goodreads API data for romance category
- **Features**: Title, author, description, ratings, reviews, publication details
- **Price Data**: Synthetically generated using probability distributions to simulate realistic pricing

### Comics & Graphic Novels
- **Source**: Goodreads API data for comics/graphic category  
- **Features**: Title, author, description, ratings, reviews, publication details
- **Price Data**: Synthetically generated using probability distributions to simulate realistic pricing

**Dataset Characteristics**: Combined dataset with comprehensive metadata, ratings, and review text for enhanced semantic search capabilities.

## 🚀 Features

### Core Capabilities
- **Semantic Search**: Find books based on meaning and context, not just keywords
- **Analytical Queries**: SQL-based filtering and aggregations (price ranges, ratings, publication years)
- **Hybrid Retrieval**: Combines vector similarity with structured data filtering
- **Conversational Interface**: Natural language interaction with context awareness
- **Real-time Processing**: Live vectorization and embedding of new data

### Technical Features
- **Multi-modal RAG**: Supports both vector and analytical retrieval methods
- **Scalable Architecture**: Modular design with separated concerns
- **Intelligent Caching**: Optimized performance through strategic caching
- **Async Processing**: Non-blocking operations for better user experience

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Vector Database**: Milvus Lite
- **Analytical Database**: DuckDB
- **Embeddings**: BGE (BAAI/bge-large-en-v1.5, BAAI/bge-small-en-v1.5)
- **LLM Integration**: OpenAI GPT models via LiteLLM
- **Workflow Engine**: CrewAI for agent orchestration
- **Data Processing**: Pandas, NumPy
- **ML Libraries**: Sentence Transformers, Scikit-learn

## 📁 Project Structure

```
├── src/
│   ├── embed_data.py          # Data embedding and vectorization
│   ├── vectordb.py            # Milvus vector database operations
│   ├── analytical_db.py       # DuckDB analytical operations
│   ├── retrieval.py           # Hybrid retrieval system
│   ├── rag.py                 # RAG pipeline implementation
│   └── sql_generator.py       # Dynamic SQL query generation
├── workflows/
│   ├── agent_workflow.py      # CrewAI agent orchestration
│   └── events.py             # Event handling system
├── config/
│   └── settings.py           # Configuration management
├── assetts/                  # System design diagrams
├── app.py                    # Main Streamlit application
├── main.py                   # Alternative entry point
├── eda.ipynb                 # Exploratory data analysis notebook
└── requirements.txt          # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd flow-dynamic-assignment
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file with your API keys
OPENAI_API_KEY=your_openai_api_key
```

4. **Run the application**
```bash
streamlit run app.py
```

## 💡 Usage

### Data Vectorization
1. Navigate to the "🔄 Vectorize" tab
2. The system will automatically process the book datasets
3. Embeddings are generated and stored in both Milvus and DuckDB

### Chatbot Interface
1. Switch to the "💬 Chatbot" tab
2. Ask natural language questions about books:
   - "Recommend romance books with high ratings"
   - "Find graphic novels under $20"
   - "What are the most popular books by Alan Moore?"
   - "Show me books published in 2017 with good reviews"

### Query Examples
- **Semantic**: "Books about love and relationships"
- **Analytical**: "Books priced between $10-$25 with rating > 4.0"
- **Hybrid**: "Popular romance novels under $15"



## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

