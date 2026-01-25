import os
import hashlib
from pathlib import Path
from typing import List, Generator

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import constants
from config.settings import settings
from utils.logging import logger


class DocumentProcessor:
    def __init__(self, embeddings: HuggingFaceEmbeddings):
        if embeddings is None:
            raise ValueError("Embeddings cannot be None when using dense retrieval.")
        self.embeddings = embeddings

        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup Qdrant client
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        self.collection_name = "docchat_documents"
        self._ensure_collection()

        # Setup vector store
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    # ---------------------- QDRANT COLLECTION ---------------------- #
    def _ensure_collection(self):
        try:
            self.qdrant_client.get_collection(self.collection_name)
        except Exception:
            logger.info("Creating Qdrant collection...")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    # ---------------------- FILE VALIDATION ---------------------- #
    def validate_files(self, files: List):
        total_size = sum(os.path.getsize(f.name) for f in files)
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(
                f"Total size exceeds {constants.MAX_TOTAL_SIZE // 1024 // 1024} MB"
            )

    # ---------------------- STREAMING PROCESS ---------------------- #
    def process(
        self, files: List, batch_size: int = 32, chunk_size: int = 1200, chunk_overlap: int = 100
    ):
        """Process files and stream chunks directly into Qdrant."""
        self.validate_files(files)

        for file in files:
            try:
                file_hash = self._file_hash(file.name)
                logger.info(f"Processing file: {file.name}")

                # Stream chunks and add to Qdrant
                for chunk_batch in self._stream_chunks(file.name, chunk_size, chunk_overlap, batch_size):
                    self.vector_store.add_documents(chunk_batch)

            except Exception as e:
                logger.error(f"Failed processing {file.name}: {str(e)}")
                continue

        logger.info("Processing complete.")

    # ---------------------- STREAMING CHUNKS ---------------------- #
    def _stream_chunks(
        self, filename: str, chunk_size: int, chunk_overlap: int, batch_size: int
    ) -> Generator[List[Document], None, None]:
        documents = self._load_file(filename)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        all_chunks = []
        for doc in documents:
            splits = splitter.split_documents([doc])
            all_chunks.extend(splits)

            while len(all_chunks) >= batch_size:
                batch, all_chunks = all_chunks[:batch_size], all_chunks[batch_size:]
                # Embed batch
                embeddings = self.embeddings.embed_documents([c.page_content for c in batch])
                for c, e in zip(batch, embeddings):
                    c.metadata["embedding"] = e
                yield batch

        if all_chunks:
            embeddings = self.embeddings.embed_documents([c.page_content for c in all_chunks])
            for c, e in zip(all_chunks, embeddings):
                c.metadata["embedding"] = e
            yield all_chunks

    # ---------------------- FILE LOADING ---------------------- #
    def _load_file(self, filename: str) -> List[Document]:
        documents = []
        if filename.endswith(".pdf"):
            with pdfplumber.open(filename) as pdf:
                for idx, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        documents.append(Document(page_content=text, metadata={"source": filename, "page": idx + 1}))
        elif filename.endswith((".txt", ".md")):
            with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            documents.append(Document(page_content=text, metadata={"source": filename}))
        else:
            logger.warning(f"Skipping unsupported file type: {filename}")
        return documents

    # ---------------------- UTILITIES ---------------------- #
    def _file_hash(self, filename: str) -> str:
        with open(filename, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()    
