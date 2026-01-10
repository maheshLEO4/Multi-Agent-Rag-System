from typing import List
import logging
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
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
            # Using .invoke() is the standard way to call any retriever.
            # It automatically handles the run_manager and internal logic.
            results = retriever.invoke(query, config={"callbacks": run_manager.get_child() if run_manager else None})

            for doc in results:
                # Use page_content hash to avoid duplicates from different retrievers
                doc_id = hash(doc.page_content)
                if doc_id not in seen:
                    seen.add(doc_id)
                    docs.append(doc)

        return docs

class RetrieverBuilder:
    def __init__(self):
        """Initialize embeddings and Qdrant client."""

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        self.collection_name = "docchat_documents"
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure Qdrant collection exists."""
        try:
            self.qdrant_client.get_collection(self.collection_name)
            logger.info("Qdrant collection exists.")
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection.")

    def build_hybrid_retriever(self, docs: List[Document]) -> BaseRetriever:
        """Build hybrid BM25 + vector retriever."""

        try:
            vector_store = Qdrant.from_documents(
                documents=docs,
                embedding=self.embeddings,
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name=self.collection_name,
                force_recreate=False,
            )

            bm25 = BM25Retriever.from_documents(docs)
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

            hybrid_retriever = HybridRetriever(
                retrievers=[bm25, vector_retriever]
            )

            logger.info("Hybrid retriever initialized successfully.")
            return hybrid_retriever

        except Exception as e:
            logger.exception("Failed to build hybrid retriever")
            raise e