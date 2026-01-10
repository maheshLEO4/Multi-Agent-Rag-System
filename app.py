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
# File hashing
# ---------------------------
def _get_file_hashes(uploaded_files: List[Path]) -> frozenset:
    """Generate SHA-256 hashes for uploaded files."""
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
        page_title="DocChat 🐥 - Powered by Docling & LangGraph",
        page_icon="📚",
        layout="wide"
    )

    # ---------------------------
    # Initialize session state
    # ---------------------------
    if 'processor' not in st.session_state:
        processor, retriever_builder, workflow = initialize_components()
        st.session_state.processor = processor
        st.session_state.retriever_builder = retriever_builder
        st.session_state.workflow = workflow
        st.session_state.file_hashes = frozenset()
        st.session_state.retriever = None
        st.session_state.uploaded_files: List[Path] = []
        st.session_state.chat_history: List[dict] = []  # {"question":..., "answer":..., "verification":...}

    # ---------------------------
    # Sidebar: Files, Examples, Info
    # ---------------------------
    with st.sidebar:
        st.markdown("## 📂 Upload & Examples")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT, MD)",
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
            st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")

        # Example data
        EXAMPLES = {
            "Google 2024 Environmental Report": {
                "question": "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 and 2022. Also retrieve regional average CFE in Asia pacific in 2023",
                "file_paths": ["examples/google-2024-environmental-report.pdf"]
            },
            "DeepSeek-R1 Technical Report": {
                "question": "Summarize DeepSeek-R1 model's performance evaluation on all coding tasks against OpenAI o1-mini model",
                "file_paths": ["examples/DeepSeek Technical Report.pdf"]
            }
        }

        example_choice = st.selectbox(
            "Select an example 🐥",
            ["Select an example..."] + list(EXAMPLES.keys())
        )

        if example_choice != "Select an example..." and st.button("Load Example 🛠️"):
            ex_data = EXAMPLES[example_choice]
            st.session_state.question_example = ex_data["question"]

            valid_files = []
            for path_str in ex_data["file_paths"]:
                path = Path(path_str)
                if path.exists():
                    valid_files.append(path)
                else:
                    st.warning(f"Example file not found: {path}")
            if valid_files:
                st.session_state.uploaded_files = valid_files
                st.success(f"Loaded example: {example_choice}")
            else:
                st.error("No valid example files found")

        st.markdown("---")
        st.markdown("### ℹ️ Information")
        st.info("""
        **Supported formats:** PDF, DOCX, TXT, MD  
        **Max file size:** 50MB  
        **Total limit:** 200MB
        """)

        # ---------------------------
        # Sidebar: Chat History & Verification
        # ---------------------------
        st.markdown("---")
        st.markdown("## 🐥 Chat History & Verification")
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"<div style='background-color:#d4edda;padding:8px;border-radius:5px;'>{chat['answer']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color:#f0f2f6;padding:5px;border-radius:5px;color:#555;'>{chat['verification']}</div>", unsafe_allow_html=True)
            st.markdown("---")

        # ---------------------------
        # Sidebar: Cleanup
        # ---------------------------
        st.markdown("## 🧹 Cleanup")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")

        if st.button("Reset All"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("All data reset!")
            st.experimental_rerun()

    # ---------------------------
    # Main Area: Chat Interface
    # ---------------------------
    st.markdown("# DocChat 🐥")
    st.markdown("Ask multiple questions continuously about your uploaded documents.")

    question = st.text_area(
        "Enter your question here...",
        height=120,
        value=st.session_state.get("question_example", "")
    )

    if st.button("🚀 Submit Question"):
        if not question.strip():
            st.error("❌ Please enter a question")
        elif not st.session_state.uploaded_files:
            st.error("❌ Please upload at least one document")
        else:
            with st.spinner("Processing..."):
                try:
                    # Check file hashes
                    current_hashes = _get_file_hashes(st.session_state.uploaded_files)
                    if st.session_state.retriever is None or current_hashes != st.session_state.file_hashes:
                        chunks = st.session_state.processor.process(st.session_state.uploaded_files)
                        # Use fixed hybrid retriever internally
                        retriever = st.session_state.retriever_builder.build_hybrid_retriever(chunks)
                        st.session_state.retriever = retriever
                        st.session_state.file_hashes = current_hashes

                    # Run workflow
                    result = st.session_state.workflow.full_pipeline(
                        question=question,
                        retriever=st.session_state.retriever
                    )

                    # Append to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result.get("draft_answer", ""),
                        "verification": result.get("verification_report", "")
                    })

                    st.success("✅ Question processed!")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Chat error: {str(e)}")

if __name__ == "__main__":
    main()
