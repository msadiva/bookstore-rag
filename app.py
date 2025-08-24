import streamlit as st
import pandas as pd
import asyncio

# ---- Your project imports ----
from src.embed_data import EmbedData
from src.vectordb import MilvusDB
from src.retrieval import Retriever
from src.rag import RAG
from src.analytical_db import AnalyticalDB
from workflows.agent_workflow import BookStoreAgentWorkflow
from config.settings import settings


# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(page_title="📚 Book RAG", layout="wide")
st.title("📚 Bookstore RAG System")
st.write("Upload books CSV → Vectorize → Chat with your book database!")


# ---------------------------------------------------------
# Session State Defaults
# ---------------------------------------------------------
if "workflow" not in st.session_state:
    st.session_state.workflow = None

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "tab1"   # default Vectorize tab

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------
# Tabs with persistence
# ---------------------------------------------------------
tab = st.radio(
    "Navigation",
    ["🔄 Vectorize", "💬 Chatbot"],
    index=0 if st.session_state.active_tab == "tab1" else 1,
    horizontal=True
)


# =========================================================
# TAB 1 – VECTORIZE
# =========================================================
if tab == "🔄 Vectorize":
    st.session_state.active_tab = "tab1"
    st.header("Vectorize Books Data")
    
    # Option to use preloaded file, upload new one, or use existing data
    data_source = st.radio(
        "Choose data source:",
        ["Use existing vectorized data", "Use preloaded data", "Upload new CSV file"],
        index=0,
        help="Select how you want to work with the data"
    )
    
    df = None
    
    if data_source == "Use existing vectorized data":
        st.info("🔗 This will connect to existing Milvus collection and DuckDB without re-vectorizing.")
        
        if st.button("🔗 Connect to Existing Data"):
            with st.spinner("Connecting to existing databases..."):
                try:
                    # Try connecting to Milvus collection directly
                    vector_db = MilvusDB(
                        collection_name=settings.collection_name,
                        vector_dim=settings.vector_dim,
                        batch_size=settings.batch_size,
                    )
                    vector_db.initialize_client()

                    if vector_db.collection is not None:  # collection already exists
                        # Create DuckDB from preloaded file
                        try:
                            analytical_db = AnalyticalDB()
                            preloaded_file_path = "data/final_vectorization_file.csv"
                            df_for_duckdb = pd.read_csv(preloaded_file_path)
                            analytical_db.create_books_table(df_for_duckdb)
                            analytical_db_available = True
                            st.success("✅ Connected to Milvus and created DuckDB from preloaded data.")
                        except Exception as e:
                            analytical_db = None
                            analytical_db_available = False
                            st.warning(f"⚠️ Connected to Milvus but couldn't create DuckDB: {e}")
                        
                        # Initialize components
                        embed_data = EmbedData(
                            embed_model_name=settings.embedding_model,
                            batch_size=settings.batch_size,
                        )
                        retriever = Retriever(vector_db=vector_db, embed_data=embed_data, top_k=settings.top_k)
                        rag_system = RAG(retriever=retriever, llm_model=settings.llm_model, temperature=settings.temperature)
                        
                        # Create workflow
                        st.session_state.workflow = BookStoreAgentWorkflow(
                            retriever=retriever, 
                            rag_system=rag_system,
                            analytical_db=analytical_db
                        )
                        
                        if analytical_db_available:
                            st.success("🎉 Ready! You can now use both semantic search and analytical queries in the Chat tab.")
                        else:
                            st.success("🎉 Ready for semantic search! For analytical queries, please vectorize new data.")
                    else:
                        st.error("❌ No existing Milvus collection found. Please vectorize data first.")
                        
                except Exception as e:
                    st.error(f"❌ Error connecting to existing data: {e}")
    
    elif data_source == "Use preloaded data":
        preloaded_file_path = "data/final_vectorization_file.csv"
        try:
            df = pd.read_csv(preloaded_file_path)
            st.success(f"✅ Loaded preloaded data: {len(df)} rows")
            st.dataframe(df.head())
        except FileNotFoundError:
            st.error(f"❌ Preloaded file not found: {preloaded_file_path}")
        except Exception as e:
            st.error(f"❌ Error loading preloaded file: {e}")
    
    else:  # Upload new CSV file
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

    # Show vectorization button only for data sources that need processing
    if data_source != "Use existing vectorized data" and df is not None and st.button("🚀 Start Vectorization"):
            with st.spinner("Vectorizing data..."):
                try:
                    texts, metadata_list = [], []

                    for _, row in df.iterrows():
                        review_text = str(row.get("review_text", ""))
                        if not review_text.strip():
                            continue

                        # Construct text for embeddings
                        text_content = (
                            f"Review: {review_text}\n\n"
                            f"Title: {row.get('title', '')}, "
                            f"Author: {row.get('name', '')}, "
                            f"Description: {row.get('description', '')}, "
                        )

                        # Metadata
                        metadata = {
                            "book_id": str(row.get("book_id", "")),
                            "title": str(row.get("title", "")),
                            "author": str(row.get("name", "")),
                            "publisher": str(row.get("publisher", "")),
                            "publication_year": int(row.get("publication_year", 0)),
                            "average_rating": float(row.get("average_rating", 0.0)),
                            "num_pages": int(row.get("num_pages", 0)),
                            "price": float(row.get("price", 0.0)),
                            "book_category": str(row.get("book_category", ""))
                        }

                        texts.append(text_content)
                        metadata_list.append(metadata)

                    # ---- Embeddings ----
                    embed_data = EmbedData(
                        embed_model_name=settings.embedding_model,
                        batch_size=settings.batch_size,
                    )
                    embed_data.embed(texts)

                    # ---- Milvus DB ----
                    vector_db = MilvusDB(
                        collection_name=settings.collection_name,
                        vector_dim=settings.vector_dim,
                        batch_size=settings.batch_size,
                    )
                    vector_db.initialize_client()
                    vector_db.create_collection()
                    vector_db.ingest_data(embed_data, metadata_list)

                    # ---- Analytical DB (DuckDB) ----
                    analytical_db = AnalyticalDB()
                    analytical_db.create_books_table(df)

                    # ---- Retriever + RAG ----
                    retriever = Retriever(vector_db=vector_db, embed_data=embed_data, top_k=settings.top_k)
                    rag_system = RAG(retriever=retriever, llm_model=settings.llm_model, temperature=settings.temperature)

                    # ---- Workflow ----
                    st.session_state.workflow = BookStoreAgentWorkflow(
                        retriever=retriever, 
                        rag_system=rag_system,
                        analytical_db=analytical_db
                    )

                    st.success("✅ Vectorization complete! Switch to Chatbot tab.")

                except Exception as e:
                    st.error(f"Error during vectorization: {e}")


# =========================================================
# TAB 2 – CHATBOT
# =========================================================
elif tab == "💬 Chatbot":
    st.session_state.active_tab = "tab2"
    st.header("Chat with Books")

    if st.session_state.workflow is None:
        st.warning("⚠️ No workflow loaded. Trying to connect to existing collection...")

        try:
            # Try connecting to Milvus collection directly
            vector_db = MilvusDB(
                collection_name=settings.collection_name,
                vector_dim=settings.vector_dim,
                batch_size=settings.batch_size,
            )
            vector_db.initialize_client()

            if vector_db.collection is not None:  # collection already exists
                embed_data = EmbedData(
                    embed_model_name=settings.embedding_model,
                    batch_size=settings.batch_size,
                )
                
                # Try to connect to existing analytical DB
                try:
                    analytical_db = AnalyticalDB()
                    # Check if books table exists
                    analytical_db.get_table_stats()
                    analytical_db_available = True
                except Exception:
                    analytical_db = None
                    analytical_db_available = False
                
                retriever = Retriever(vector_db=vector_db, embed_data=embed_data, top_k=settings.top_k)
                rag_system = RAG(retriever=retriever, llm_model=settings.llm_model, temperature=settings.temperature)
                
                st.session_state.workflow = BookStoreAgentWorkflow(
                    retriever=retriever, 
                    rag_system=rag_system,
                    analytical_db=analytical_db
                )
                
                if analytical_db_available:
                    st.success("✅ Connected to existing collection and analytical database. You can now chat and run analytical queries.")
                else:
                    st.success("✅ Connected to existing collection. Analytical queries not available - please re-vectorize data.")
                    st.warning("⚠️ For analytical queries (counts, averages, etc.), please upload and vectorize data again.")
            else:
                st.warning("⚠️ No collection found. Please vectorize in Tab 1.")

        except Exception as e:
            st.error(f"Error initializing RAG pipeline: {e}")

    if st.session_state.workflow:
        # Chat input
        if query := st.chat_input("Ask something about the books..."):
            # Add user message to history and rerun to display it
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.rerun()
        
        # Display chat history (including the processing state)
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
        
        # If the last message is from user and no assistant response yet, generate response
        if (st.session_state.chat_history and 
            st.session_state.chat_history[-1]["role"] == "user"):
            
            last_user_query = st.session_state.chat_history[-1]["content"]
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        print(f"🚀 CALLING WORKFLOW WITH QUERY: {last_user_query}")
                        result = asyncio.run(st.session_state.workflow.run_workflow(last_user_query))
                        print(f"🎯 APP RECEIVED RESULT: {result}")
                        answer = result.get("answer", "⚠️ No answer available.")
                        
                        st.write(answer)
                        
                        # Add assistant response to history
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        st.rerun()
                        
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                        st.rerun()
        
        # Add clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
