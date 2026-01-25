from typing import List
import logging
import os

from langchain_huggingface import HuggingFaceEmbeddings
# FIX: Use the new dedicated Qdrant integration
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining multiple retrievers safely."""
    retrievers: List[BaseRetriever]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs: List[Document] = []
        seen = set()

        for retriever in self.retrievers:
            # Use invoke to let LangChain handle internal run_manager logic
            results = retriever.invoke(
                query, 
                config={"callbacks": run_manager.get_child() if run_manager else None}
            )

            for doc in results:
                doc_id = hash(doc.page_content)
                if doc_id not in seen:
                    seen.add(doc_id)
                    docs.append(doc)
        return docs

class RetrieverBuilder:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = os.getenv("QDRANT_API_KEY")
        
        self.qdrant_client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )

        self.collection_name = "docchat_documents"
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.qdrant_client.get_collection(self.collection_name)
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def build_hybrid_retriever(self, docs: List[Document]) -> BaseRetriever:
        try:
            # Use the new QdrantVectorStore class
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            
            # Clean collection and add new docs
            vector_store.add_documents(docs)

            bm25 = BM25Retriever.from_documents(docs)
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

            return HybridRetriever(retrievers=[bm25, vector_retriever])

        except Exception as e:
            logger.exception("Failed to build hybrid retriever")
            raise e    find erroe 
