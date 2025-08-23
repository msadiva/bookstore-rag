import streamlit as st
import pandas as pd

# ---- Your project imports ----
from src.embed_data import EmbedData
from src.vectordb import MilvusDB
from src.retrieval import Retriever
from src.rag import RAG
from workflows.agent_workflow import BookStoreAgentWorkflow
from config.settings import settings
import asyncio 


# ---- UI ----
st.set_page_config(page_title="📚 Book RAG", layout="wide")

st.title("📚 Bookstore RAG System")
st.write("Upload books CSV → Vectorize → Chat with your book database!")

# ---- Tabs ----
tab1, tab2 = st.tabs(["🔄 Vectorize", "💬 Chatbot"])

# ==========================================================
# TAB 1 – VECTORIZE
# ==========================================================
with tab1:
    st.header("Upload and Vectorize Books")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.button("🚀 Start Vectorization"):
            with st.spinner("Vectorizing data..."):
                try:
                    texts = []        # for embeddings
                    metadata_list = []  # for Milvus
                    for _, row in df.iterrows():
                        review_text = str(row.get("review_text", ""))
                        if not review_text.strip():
                            continue  # skip empty rows

                        # Construct natural text for embedding
                        text_content = (
                            f"Review: {review_text}\n\n"
                            f"Title: {row.get('title', '')}, "
                            f"Author: {row.get('name', '')}, "
                            f"Description: {row.get('description', '')}, "
                        )

                        # Metadata for filtering
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

                    # ---- Milvus Vector DB ----
                    vector_db = MilvusDB(
                        collection_name=f"{settings.collection_name}",
                        vector_dim=settings.vector_dim,
                        batch_size=settings.batch_size,
                    )
                    vector_db.initialize_client()
                    vector_db.create_collection()
                    vector_db.ingest_data(embed_data, metadata_list)
                    st.session_state.vector_db = vector_db

                    # ---- Retriever + RAG ----
                    retriever = Retriever(
                        vector_db=vector_db,
                        embed_data=embed_data,
                        top_k=settings.top_k,
                    )
                    rag_system = RAG(
                        retriever=retriever,
                        llm_model=settings.llm_model,
                        temperature=settings.temperature                    
                        )

                    # ---- Workflow ----
                    workflow = BookStoreAgentWorkflow(
                        retriever=retriever,
                        rag_system=rag_system,
                    )
                    st.session_state.workflow = workflow

                    st.success("✅ Vectorization complete! Switch to Chatbot tab.")

                except Exception as e:
                    st.error(f"Error during vectorization: {e}")

# ==========================================================
# TAB 2 – CHATBOT
# ==========================================================
with tab2:
    st.header("Chat with Books")

    if "workflow" not in st.session_state:
        st.warning("⚠️ Please upload & vectorize data first (Tab 1).")
    else:
        query = st.text_input("Ask something about the books:")
        if st.button("💬 Get Answer"):
            with st.spinner("Thinking..."):
                try:
                    result = asyncio.run(st.session_state.workflow.run_workflow(query))
                    st.subheader("Answer")
                    st.write(result.get("answer", "⚠️ No answer."))
                except Exception as e:
                    st.error(f"Error during chat: {e}")
