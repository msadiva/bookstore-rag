import streamlit as st
import pandas as pd
import asyncio

# ---- Your project imports ----
from src.embed_data import EmbedData
from src.vectordb import MilvusDB
from src.retrieval import Retriever
from src.rag import RAG
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
    st.header("Upload and Vectorize Books")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.button("🚀 Start Vectorization"):
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
                            "price": float(row.get("price", 0.0))
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

                    # ---- Retriever + RAG ----
                    retriever = Retriever(vector_db=vector_db, embed_data=embed_data, top_k=settings.top_k)
                    rag_system = RAG(retriever=retriever, llm_model=settings.llm_model, temperature=settings.temperature)

                    # ---- Workflow ----
                    st.session_state.workflow = BookStoreAgentWorkflow(retriever=retriever, rag_system=rag_system)

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
                retriever = Retriever(vector_db=vector_db, embed_data=embed_data, top_k=settings.top_k)
                rag_system = RAG(retriever=retriever, llm_model=settings.llm_model, temperature=settings.temperature)
                st.session_state.workflow = BookStoreAgentWorkflow(retriever=retriever, rag_system=rag_system)
                st.success("✅ Connected to existing collection. You can now chat.")
            else:
                st.warning("⚠️ No collection found. Please vectorize in Tab 1.")

        except Exception as e:
            st.error(f"Error initializing RAG pipeline: {e}")

    if st.session_state.workflow:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
        
        # Chat input
        if query := st.chat_input("Ask something about the books..."):
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Display user message immediately
            with st.chat_message("user"):
                st.write(query)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = asyncio.run(st.session_state.workflow.run_workflow(query))
                        answer = result.get("answer", "⚠️ No answer available.")
                        
                        st.write(answer)
                        
                        # Add assistant response to history
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        
        # Add clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
