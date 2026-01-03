import streamlit as st
import hashlib
import os
from typing import List, Dict

from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from utils.logging import logger


# -------------------- INIT --------------------

@st.cache_resource
def initialize_components():
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow()
    return processor, retriever_builder, workflow


# -------------------- UTILS --------------------

def _get_file_hashes(file_paths: List[str]) -> frozenset:
    hashes = set()
    for path in file_paths:
        try:
            with open(path, "rb") as f:
                hashes.add(hashlib.sha256(f.read()).hexdigest())
        except Exception:
            continue
    return frozenset(hashes)


# -------------------- MAIN --------------------

def main():
    st.set_page_config(
        page_title="DocChat 🐥",
        page_icon="📚",
        layout="wide",
    )

    # ---------- SESSION STATE ----------
    if "processor" not in st.session_state:
        processor, retriever_builder, workflow = initialize_components()
        st.session_state.processor = processor
        st.session_state.retriever_builder = retriever_builder
        st.session_state.workflow = workflow

        st.session_state.uploaded_files: List[str] = []
        st.session_state.file_hashes = frozenset()
        st.session_state.retriever = None

        st.session_state.chat_history: List[Dict] = []

    # ================= SIDEBAR =================
    with st.sidebar:
        st.markdown("## 📂 Document Setup")

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            temp_files = []
            for uf in uploaded_files:
                temp_path = f"temp_{uf.name}"
                with open(temp_path, "wb") as f:
                    f.write(uf.getbuffer())
                temp_files.append(temp_path)

            st.session_state.uploaded_files = temp_files
            st.success(f"{len(temp_files)} file(s) uploaded")

        st.markdown("---")

        if st.button("🧹 Reset Documents", use_container_width=True):
            st.session_state.uploaded_files = []
            st.session_state.file_hashes = frozenset()
            st.session_state.retriever = None
            st.session_state.chat_history = []
            st.success("Documents reset")
            st.rerun()

        if st.button("🧠 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat cleared")
            st.rerun()

        st.markdown("---")
        st.info(
            """
            **Tech Stack**
            - Docling 🐥
            - LangGraph
            - Hybrid Retriever (BM25 + Vector)
            - Qdrant Cloud
            """
        )

    # ================= MAIN CHAT =================

    st.markdown("## 🐥 DocChat")

    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        if not st.session_state.uploaded_files:
            st.error("Please upload documents first.")
            return

        # Add user message
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    current_hashes = _get_file_hashes(
                        st.session_state.uploaded_files
                    )

                    # Build retriever only if needed
                    if (
                        st.session_state.retriever is None
                        or current_hashes != st.session_state.file_hashes
                    ):
                        logger.info("Building retriever...")
                        docs = st.session_state.processor.process(
                            st.session_state.uploaded_files
                        )
                        retriever = (
                            st.session_state.retriever_builder
                            .build_hybrid_retriever(docs)
                        )

                        st.session_state.retriever = retriever
                        st.session_state.file_hashes = current_hashes

                    # Run workflow
                    result = st.session_state.workflow.full_pipeline(
                        question=question,
                        retriever=st.session_state.retriever,
                    )

                    answer = result["draft_answer"]
                    verification = result["verification_report"]

                    st.markdown(answer)

                    with st.expander("✅ Verification"):
                        st.markdown(verification)

                    # Save assistant message
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    st.error("Something went wrong. Check logs.")

    # Footer
    st.markdown(
        """
        <div style="text-align:center;color:#777;margin-top:20px;">
        Powered by Docling 🐥 · LangGraph · Qdrant
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
