import streamlit as st
import hashlib
import os
from typing import List
from pathlib import Path

from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from config import constants
from utils.logging import logger

# Initialize components
@st.cache_resource
def initialize_components():
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow()
    return processor, retriever_builder, workflow

def main():
    st.set_page_config(
        page_title="DocChat 🐥 - Powered by Docling & LangGraph",
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #FFD700;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.2em;
        color: #FFD700;
        margin-bottom: 15px;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">DocChat 🐥</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Powered by Docling 🐥 and LangGraph</h3>', unsafe_allow_html=True)
    
    # How it works section
    with st.expander("✨ How it works", expanded=True):
        st.markdown("""
        1. **📤 Upload** your document(s) (PDF, DOCX, TXT, MD)
        2. **📝 Enter** your question
        3. **🚀 Submit** to get AI-powered answers
        4. **✅ Verification** report shows answer accuracy
        
        **Note:** DocChat uses:
        - **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
        - **LLM:** llama-3.1-8b-instruct (Groq)
        - **Vector DB:** Qdrant Cloud
        - **Document Processing:** Docling
        - **Workflow:** LangGraph multi-agent system
        """)
    
    # Initialize session state
    if 'processor' not in st.session_state:
        processor, retriever_builder, workflow = initialize_components()
        st.session_state.processor = processor
        st.session_state.retriever_builder = retriever_builder
        st.session_state.workflow = workflow
        st.session_state.file_hashes = frozenset()
        st.session_state.retriever = None
        st.session_state.uploaded_files = []
        st.session_state.question = ""
        st.session_state.answer = ""
        st.session_state.verification = ""
    
    # Sidebar for examples
    with st.sidebar:
        st.markdown("### 📂 Examples")
        
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
        
        if example_choice != "Select an example...":
            if st.button("Load Example 🛠️"):
                ex_data = EXAMPLES[example_choice]
                st.session_state.question = ex_data["question"]
                
                # Check if example files exist
                valid_files = []
                for path in ex_data["file_paths"]:
                    if os.path.exists(path):
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
        **Supported formats:**
        - PDF (.pdf)
        - Word (.docx)
        - Text (.txt)
        - Markdown (.md)
        
        **Max file size:** 50MB
        **Total limit:** 200MB
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📄 Upload Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="Upload one or more documents"
        )
        
        if uploaded_files:
            # Save uploaded files temporarily
            temp_files = []
            for uploaded_file in uploaded_files:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_files.append(temp_path)
            
            st.session_state.uploaded_files = temp_files
            st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")
        
        st.markdown("### 📝 Enter Question")
        question = st.text_area(
            "Ask a question about your documents",
            value=st.session_state.question,
            height=100,
            placeholder="Enter your question here..."
        )
        st.session_state.question = question
        
        # Submit button
        if st.button("🚀 Submit", type="primary", use_container_width=True):
            if not question.strip():
                st.error("❌ Please enter a question")
            elif not st.session_state.uploaded_files:
                st.error("❌ Please upload at least one document")
            else:
                with st.spinner("Processing..."):
                    try:
                        # Get current file hashes
                        current_hashes = _get_file_hashes(st.session_state.uploaded_files)
                        
                        # Rebuild retriever if files changed
                        if (st.session_state.retriever is None or 
                            current_hashes != st.session_state.file_hashes):
                            
                            with st.status("Processing documents...", expanded=True) as status:
                                st.write("📊 Extracting text with Docling...")
                                chunks = st.session_state.processor.process(st.session_state.uploaded_files)
                                
                                st.write("🔍 Building hybrid retriever...")
                                retriever = st.session_state.retriever_builder.build_hybrid_retriever(chunks)
                                
                                # Update session state
                                st.session_state.file_hashes = current_hashes
                                st.session_state.retriever = retriever
                                
                                status.update(label="✅ Documents processed!", state="complete")
                        
                        # Run workflow
                        with st.status("Running multi-agent workflow...", expanded=True) as status:
                            st.write("🤖 Checking relevance...")
                            st.write("🧠 Researching answer...")
                            st.write("✅ Verifying accuracy...")
                            
                            result = st.session_state.workflow.full_pipeline(
                                question=question,
                                retriever=st.session_state.retriever
                            )
                            
                            st.session_state.answer = result["draft_answer"]
                            st.session_state.verification = result["verification_report"]
                            
                            status.update(label="✅ Analysis complete!", state="complete")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        logger.error(f"Processing error: {str(e)}")
    
    with col2:
        st.markdown("### 🐥 Answer")
        
        if st.session_state.answer:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown(st.session_state.answer)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Upload documents and submit a question to see the answer here")
        
        st.markdown("### ✅ Verification Report")
        
        if st.session_state.verification:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(st.session_state.verification)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Verification report will appear here after submission")
    
    # Cleanup section
    with st.expander("🧹 Cleanup Options"):
        col_clean1, col_clean2 = st.columns(2)
        
        with col_clean1:
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.answer = ""
                st.session_state.verification = ""
                st.success("Chat history cleared!")
                st.rerun()
        
        with col_clean2:
            if st.button("Reset All", use_container_width=True, type="secondary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("All data reset!")
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p>Powered by: 
    <strong>Docling</strong> 🐥 | 
    <strong>LangGraph</strong> | 
    <strong>sentence-transformers/all-MiniLM-L6-v2</strong> | 
    <strong>llama-3.1-8b-instruct</strong> | 
    <strong>Qdrant</strong>
    </p>
    </div>
    """, unsafe_allow_html=True)

def _get_file_hashes(uploaded_files: List[str]) -> frozenset:
    """Generate SHA-256 hashes for uploaded files."""
    hashes = set()
    for file_path in uploaded_files:
        try:
            with open(file_path, "rb") as f:
                hashes.add(hashlib.sha256(f.read()).hexdigest())
        except:
            continue
    return frozenset(hashes)

if __name__ == "__main__":
    main()