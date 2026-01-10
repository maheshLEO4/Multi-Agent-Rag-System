import streamlit as st
import hashlib
import os
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
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow()
    return processor, retriever_builder, workflow

# ---------------------------
# Helper functions
# ---------------------------
def _get_file_hashes(uploaded_files: List[Path]) -> frozenset:
    hashes = set()
    for file_path in uploaded_files:
        try:
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

    # Initialize session state
    if 'processor' not in st.session_state:
        processor, retriever_builder, workflow = initialize_components()
        st.session_state.processor = processor
        st.session_state.retriever_builder = retriever_builder
        st.session_state.workflow = workflow
        st.session_state.file_hashes = frozenset()
        st.session_state.retriever = None
        st.session_state.uploaded_files: List[Path] = []
        st.session_state.messages = []  # Chat history for UI

    # ---------------------------
    # SIDEBAR: Management & Metrics
    # ---------------------------
    with st.sidebar:
        st.title("⚙️ Control Panel")
        
        # 1. File Upload Section
        st.markdown("### 📂 Documents")
        uploaded_files = st.file_uploader(
            "Upload sources",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True
        )

        if uploaded_files:
            temp_files = []
            for uploaded_file in uploaded_files:
                temp_path = Path(f"temp_{uploaded_file.name}")
                temp_path.write_bytes(uploaded_file.getbuffer())
                temp_files.append(temp_path)
            st.session_state.uploaded_files = temp_files
            st.success(f"Loaded {len(uploaded_files)} files")

        # 2. Example Queries
        st.markdown("### 💡 Examples")
        EXAMPLES = {
            "Google Sustainability": "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 and 2022.",
            "DeepSeek-R1 Coding": "Summarize DeepSeek-R1 model's performance on coding tasks vs OpenAI o1-mini."
        }
        
        example_choice = st.selectbox("Quick Start", ["Select an example..."] + list(EXAMPLES.keys()))
        if example_choice != "Select an example...":
            st.info(EXAMPLES[example_choice])

        # 3. System Metrics (Placeholders for real-time stats)
        st.markdown("### 📊 System Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", "1.2k" if st.session_state.retriever else "0")
        with col2:
            st.metric("Latency", "1.2s")

        st.markdown("---")
        if st.button("🧹 Reset System", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ---------------------------
    # MAIN PAGE: Chat UI
    # ---------------------------
    st.title("DocChat 🐥")
    st.caption("Multi-Agent RAG System powered by Docling & LangGraph")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "verification" in message:
                with st.expander("🔍 Verification Report"):
                    st.caption(message["verification"])

    # React to user input
    if prompt := st.chat_input("What would you like to know about your documents?"):
        
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not st.session_state.uploaded_files:
            with st.chat_message("assistant"):
                st.error("Please upload documents in the sidebar first!")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Agents are thinking..."):
                    try:
                        # 1. Update Retriever if files changed
                        current_hashes = _get_file_hashes(st.session_state.uploaded_files)
                        if st.session_state.retriever is None or current_hashes != st.session_state.file_hashes:
                            chunks = st.session_state.processor.process(st.session_state.uploaded_files)
                            retriever = st.session_state.retriever_builder.build_hybrid_retriever(chunks)
                            st.session_state.retriever = retriever
                            st.session_state.file_hashes = current_hashes

                        # 2. Run Workflow
                        result = st.session_state.workflow.full_pipeline(
                            question=prompt,
                            retriever=st.session_state.retriever
                        )
                        
                        full_response = result.get("draft_answer", "I couldn't find an answer.")
                        verification = result.get("verification_report", "No verification performed.")

                        # 3. Show Response
                        st.markdown(full_response)
                        with st.expander("🔍 Verification Report"):
                            st.caption(verification)

                        # 4. Save to history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response,
                            "verification": verification
                        })

                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        logger.error(error_msg)

if __name__ == "__main__":
    main()