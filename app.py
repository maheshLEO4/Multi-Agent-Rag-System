import streamlit as st
import logging
from pathlib import Path

from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from utils.logging import logger as app_logger

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Multi-Agent RAG System", layout="wide")
st.title("Multi-Agent RAG System")

# ---------------------------
# Session State Init
# ---------------------------
if "workflow" not in st.session_state:
    st.session_state.workflow = AgentWorkflow()
if "documents" not in st.session_state:
    st.session_state.documents = []

# ---------------------------
# File Upload Section
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload PDF / DOCX / TXT files", 
    accept_multiple_files=True, 
    type=["pdf", "docx", "txt"]
)

if uploaded_files:
    st.session_state.documents = []
    processor = DocumentProcessor()
    for file in uploaded_files:
        st.info(f"Processing file: {file.name}")
        docs = processor.process(file)
        st.session_state.documents.extend(docs)
    st.success(f"Processed {len(st.session_state.documents)} documents")

# ---------------------------
# Question Input Section
# ---------------------------
question = st.text_input("Enter your question here:")

# ---------------------------
# Build Retriever
# ---------------------------
if st.session_state.documents:
    retriever = RetrieverBuilder.build_hybrid_retriever(
        documents=st.session_state.documents
    )
else:
    retriever = None

# ---------------------------
# Ask Button
# ---------------------------
if st.button("Ask") and question:
    if not retriever:
        st.error("No documents uploaded to create retriever!")
    else:
        try:
            # 🔹 Use full_pipeline with proper retriever
            result = st.session_state.workflow.full_pipeline(
                question=question,
                retriever=retriever
            )

            st.subheader("Draft Answer")
            st.write(result.get("draft_answer", ""))

            st.subheader("Verification Report")
            st.write(result.get("verification_report", ""))

        except Exception as e:
            st.error(f"Workflow execution failed: {e}")
            app_logger.exception("Workflow execution failed")
