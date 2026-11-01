import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
import tempfile
import hashlib

import streamlit as st

# Load cloud secrets before importing KnowledgeBase.
# Locally, llm_service.py will load credentials from .env.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    if "GROQ_MODEL" in st.secrets:
        os.environ["GROQ_MODEL"] = st.secrets["GROQ_MODEL"]
except Exception:
    pass

from backend.services.knowledge_base import KnowledgeBase


# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="KnowCache | Enterprise AI Knowledge Base",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 KnowCache")
st.caption(
    "Intelligent Enterprise Knowledge Base "
    "| RAG + Semantic Caching"
)

st.markdown(
    """
    Upload your company documents and ask questions in natural language.
    KnowCache retrieves relevant information and generates document-grounded answers.
    """
)


# -------------------------------
# SESSION STATE
# -------------------------------

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = KnowledgeBase()

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

if "messages" not in st.session_state:
    st.session_state.messages = []


kb = st.session_state.knowledge_base


# -------------------------------
# SIDEBAR: DOCUMENT UPLOAD
# -------------------------------

with st.sidebar:

    st.header("📄 Document Manager")

    uploaded_files = st.file_uploader(
        "Upload company PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents", use_container_width=True):

        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:

            newly_added = 0

            for uploaded_file in uploaded_files:

                file_bytes = uploaded_file.getvalue()

                # Identify duplicate uploads by file content
                file_hash = hashlib.sha256(file_bytes).hexdigest()

                if file_hash in st.session_state.indexed_files:
                    st.info(f"{uploaded_file.name} is already indexed.")
                    continue

                temp_path = None

                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(file_bytes)
                        temp_path = temp_file.name

                    with st.spinner(f"Processing {uploaded_file.name}..."):

                        chunk_count = kb.add_pdf(temp_path)

                    if chunk_count > 0:
                        st.session_state.indexed_files.add(file_hash)
                        newly_added += 1

                        st.success(
                            f"{uploaded_file.name}: "
                            f"{chunk_count} chunks indexed"
                        )
                    else:
                        st.warning(
                            f"No readable text found in {uploaded_file.name}."
                        )

                except Exception as error:
                    st.error(f"Could not process {uploaded_file.name}: {error}")

                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            # New documents can change answers.
            # Clear old cached answers to avoid stale responses.
            if newly_added:
                kb.semantic_cache.cache.clear()
                kb.semantic_cache.hits = 0
                kb.semantic_cache.misses = 0

                st.session_state.messages = []

                st.success("Document indexing completed.")


    st.divider()

    st.subheader("How it works")

    st.markdown(
        """
        1. Upload PDF
        2. Extract and chunk text
        3. Generate embeddings
        4. Retrieve relevant chunks
        5. Generate answer with Groq
        6. Reuse safe semantic cache matches
        """
    )

    st.caption("Demo version: data is stored in memory.")


# -------------------------------
# MAIN: CACHE ANALYTICS
# -------------------------------

stats = kb.semantic_cache.get_stats()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Cache Hits", stats["hits"])
col2.metric("Cache Misses", stats["misses"])
col3.metric("Total Requests", stats["total_requests"])
col4.metric("Cache Hit Rate", f'{stats["hit_rate_percent"]}%')


st.divider()


# -------------------------------
# MAIN: QUESTION ANSWERING
# -------------------------------

st.subheader("💬 Ask your documents")

# Display conversation history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            if message.get("cache_hit") is not None:

                if message["cache_hit"]:
                    st.success("Semantic Cache HIT")
                else:
                    st.info("Cache MISS — RAG retrieval and LLM generation")

                st.caption(
                    f'Latency: {message["latency_ms"]} ms'
                )

            if message.get("sources"):

                with st.expander("View source references"):

                    for source in message["sources"]:
                        st.write(
                            f'Page {source["page"]} '
                            f'| Similarity: {source["similarity"]}'
                        )


question = st.chat_input(
    "Ask something about your uploaded documents..."
)

if question:

    if not st.session_state.indexed_files:
        st.warning("Upload and process a PDF before asking questions.")

    else:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:
                with st.spinner("Analyzing your documents..."):

                    result = kb.ask_question(question)

                st.markdown(result["answer"])

                if result["cache_hit"]:
                    st.success("⚡ Semantic Cache HIT")
                else:
                    st.info("🔍 Cache MISS — RAG + LLM executed")

                st.caption(
                    f'Latency: {result["latency_ms"]} ms'
                )

                if result["sources"]:

                    with st.expander("View source references"):

                        for source in result["sources"]:
                            st.write(
                                f'Page {source["page"]} '
                                f'| Similarity: {source["similarity"]}'
                            )

                else:
                    st.caption("No document sources were retrieved.")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "cache_hit": result["cache_hit"],
                    "latency_ms": result["latency_ms"],
                    "sources": result["sources"]
                })

                # Refresh analytics after every question
                st.rerun()

            except Exception as error:

                st.error(f"Something went wrong: {error}")
