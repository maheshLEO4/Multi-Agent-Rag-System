import os
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import constants
from config.settings import settings
from utils.logging import logger


class DocumentProcessor:
    def __init__(self, embeddings=None):
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings  # HuggingFace / OpenAI embeddings instance

    # ------------------------ FILE VALIDATION ------------------------ #
    def validate_files(self, files: List) -> None:
        total_size = sum(os.path.getsize(f.name) for f in files)
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(
                f"Total size exceeds {constants.MAX_TOTAL_SIZE // 1024 // 1024}MB limit"
            )

    # ------------------------ PROCESSING ------------------------ #
    def process(self, files: List, batch_size: int = 32, max_chunks: int = 300) -> List:
        """
        Process files with caching and Streamlit-safe embedding batching.
        Returns list of embedded documents.
        """
        self.validate_files(files)
        all_chunks = []
        seen_hashes = set()

        for file in files:
            try:
                with open(file.name, "rb") as f:
                    file_hash = self._generate_hash(f.read())

                chunk_cache_path = self.cache_dir / f"{file_hash}_chunks.pkl"
                embed_cache_path = self.cache_dir / f"{file_hash}_embeddings.pkl"

                # ---------------- LOAD CHUNKS FROM CACHE ---------------- #
                if self._is_cache_valid(chunk_cache_path):
                    logger.info(f"Loading chunks from cache: {file.name}")
                    chunks = self._load_from_cache(chunk_cache_path)
                else:
                    logger.info(f"Processing and caching: {file.name}")
                    chunks = self._process_file(file)
                    chunks = self._split_documents(chunks)

                    if not chunks:
                        logger.warning(f"No content extracted from {file.name}")
                        continue

                    self._save_to_cache(chunks, chunk_cache_path)

                # ---------------- DEDUPLICATE ---------------- #
                unique_chunks = []
                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode())
                    if chunk_hash not in seen_hashes:
                        unique_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)
                chunks = unique_chunks

                # ---------------- EMBEDDINGS ---------------- #
                if self.embeddings:
                    if self._is_cache_valid(embed_cache_path):
                        logger.info(f"Loading embeddings from cache: {file.name}")
                        chunks = self._load_from_cache(embed_cache_path)
                    else:
                        logger.info(f"Creating embeddings for: {file.name}")
                        chunks = self._embed_chunks_in_batches(chunks, batch_size)
                        self._save_to_cache(chunks, embed_cache_path)

                all_chunks.extend(chunks)

            except Exception as e:
                logger.error(f"Failed to process {file.name}: {str(e)}")
                continue

        if len(all_chunks) > max_chunks:
            logger.warning(
                f"Limiting total chunks to {max_chunks} to avoid Streamlit OOM"
            )
            all_chunks = all_chunks[:max_chunks]

        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    # ------------------------ FILE PROCESSING ------------------------ #
    def _process_file(self, file) -> List[Document]:
        if file.name.endswith(".pdf"):
            documents = []
            with pdfplumber.open(file.name) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={"source": file.name, "page": page_idx + 1},
                            )
                        )
            return documents

        elif file.name.endswith((".txt", ".md")):
            with open(file.name, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": file.name})]

        else:
            logger.warning(f"Skipping unsupported file type: {file.name}")
            return []

    # ------------------------ SPLITTING ------------------------ #
    def _split_documents(self, documents: List[Document], chunk_size=1200, chunk_overlap=100) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        all_chunks = []
        for doc in documents:
            splits = splitter.split_documents([doc])
            all_chunks.extend(splits)
        return all_chunks

    # ------------------------ BATCH EMBEDDING ------------------------ #
    def _embed_chunks_in_batches(self, chunks: List[Document], batch_size: int) -> List[Document]:
        if not self.embeddings:
            return chunks

        embedded_chunks = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            # Embed and attach embedding to metadata (optional)
            embeddings = self.embeddings.embed_documents([c.page_content for c in batch])
            for c, e in zip(batch, embeddings):
                c.metadata["embedding"] = e
            embedded_chunks.extend(batch)
        return embedded_chunks

    # ------------------------ UTILITIES ------------------------ #
    def _generate_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _save_to_cache(self, data: List, cache_path: Path):
        with open(cache_path, "wb") as f:
            pickle.dump({"timestamp": datetime.now().timestamp(), "chunks": data}, f)

    def _load_from_cache(self, cache_path: Path) -> List:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]

    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)
