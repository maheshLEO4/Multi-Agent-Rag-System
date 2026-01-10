import streamlit as st
import hashlib
import os
import logging
from typing import List
from pathlib import Path

from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from utils.logging import logger

# ---------------------------
# Initialize components (cached)
# ---------------------------
@st.cache_resource
def initialize_components():
    """Initializes the heavy AI components once and caches them."""
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow()
    return processor, retriever_builder, workflow

# ---------------------------
# Helper functions
# ---------------------------
def _get_file_hashes(uploaded_files: List[Path]) -> frozenset:
    """Generate SHA-256 hashes to detect if document set has changed."""
    hashes = set()
    for file_path in uploaded_files:
        try:
            if file_path.exists():
                with open(file_path, "rb") as f:
                    hashes.add(hashlib.sha256(f.read()).hexdigest())
        except Exception as e:
            logger.error(f"Error hashing {file_path}: {e}")
            continue
    return frozenset(hashes)

# ---------------------------
# Main App
# ---------------------------
def main():
    st.set_page_config(
        page_title="DocChat 🐥",
        page_icon="🐥",
        layout="wide"
    )

    # --- 1. SAFE STATE INITIALIZATION ---
    # We initialize these immediately so the UI components always find them
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "file_hashes" not in st.session_state:
        st.session_state.file_hashes = frozenset()
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    # --- 2. COMPONENT INITIALIZATION ---
    if 'processor' not in st.session_state:
        try:
            processor, retriever_builder, workflow = initialize_components()
            st.session_state.processor = processor
            st.session_state.retriever_builder = retriever_builder
            st.session_state.workflow = workflow
        except Exception as e:
            st.error(f"Failed to initialize AI components: {e}")
            return

    # --- 3. SIDEBAR: Control Panel ---
    with st.sidebar:
        st.title("⚙️ Control Panel")
        st.markdown("---")
        
        # File Upload Section
        st.subheader("📂 Document Management")
        uploaded_files = st.file_uploader(
            "Upload sources",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="Docling will process these into searchable chunks."
        )

        if uploaded_files:
            temp_files = []
            for uploaded_file in uploaded_files:
                temp_path = Path(f"temp_{uploaded_file.name}")
                temp_path.write_bytes(uploaded_file.getbuffer())
                temp_files.append(temp_path)
            st.session_state.uploaded_files = temp_files
            st.success(f"Registered {len(uploaded_files)} files")

        # Example Section
        st.markdown("---")
        st.subheader("💡 Quick Examples")
        EXAMPLES = {
            "Google Sustainability": "Retrieve data center PUE values in Singapore for 2019 and 2022.",
            "DeepSeek-R1": "Compare DeepSeek-R1 coding performance against OpenAI o1-mini."
        }
        
        example_choice = st.selectbox("Load Example Query", ["None"] + list(EXAMPLES.keys()))
        if example_choice != "None":
            st.info(EXAMPLES[example_choice])

        # Metrics/Status
        st.markdown("---")
        st.subheader("📊 System Status")
        status_color = "🟢" if st.session_state.retriever else "⚪"
        st.write(f"{status_color} Retriever: {'Active' if st.session_state.retriever else 'Idle'}")
        
        if st.button("🧹 Reset All History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.retriever = None
            st.session_state.file_hashes = frozenset()
            st.rerun()

    # --- 4. MAIN INTERFACE: Chat Experience ---
    st.title("DocChat 🐥")
    st.caption("Advanced RAG using LangGraph Agents & Docling Intelligence")

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "verification" in message and message["verification"]:
                with st.expander("🔍 Verification Evidence"):
                    st.caption(message["verification"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        
        # Show user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process logic
        if not st.session_state.uploaded_files:
            with st.chat_message("assistant"):
                st.warning("⚠️ Please upload documents in the sidebar before asking questions.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("🤖 Agents analyzing documents..."):
                    try:
                        # Update Retriever if files changed or not yet built
                        current_hashes = _get_file_hashes(st.session_state.uploaded_files)
                        if st.session_state.retriever is None or current_hashes != st.session_state.file_hashes:
                            with st.status("Ingesting documents...", expanded=False):
                                chunks = st.session_state.processor.process(st.session_state.uploaded_files)
                                st.write(f"Processed {len(chunks)} chunks.")
                                retriever = st.session_state.retriever_builder.build_hybrid_retriever(chunks)
                                st.session_state.retriever = retriever
                                st.session_state.file_hashes = current_hashes

                        # Execute the Agentic Workflow
                        result = st.session_state.workflow.full_pipeline(
                            question=prompt,
                            retriever=st.session_state.retriever
                        )
                        
                        answer = result.get("draft_answer", "I'm sorry, I couldn't find relevant information.")
                        verification = result.get("verification_report", "")

                        # Render Results
                        st.markdown(answer)
                        if verification:
                            with st.expander("🔍 Verification Evidence"):
                                st.caption(verification)

                        # Store in history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "verification": verification
                        })

                    except Exception as e:
                        logger.exception("Chat Workflow Error")
                        st.error(f"Execution Error: {str(e)}")

if __name__ == "__main__":
    main()