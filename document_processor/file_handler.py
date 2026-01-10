import os
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


from config import constants
from config.settings import settings
from utils.logging import logger


class DocumentProcessor:
    def __init__(self):
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def validate_files(self, files: List) -> None:
        """Validate the total size of the uploaded files."""
        total_size = sum(os.path.getsize(f.name) for f in files)
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(
                f"Total size exceeds {constants.MAX_TOTAL_SIZE // 1024 // 1024}MB limit"
            )

    def process(self, files: List) -> List:
        """Process files with caching for subsequent queries"""
        self.validate_files(files)

        all_chunks = []
        seen_hashes = set()

        for file in files:
            try:
                # Generate content-based hash for caching
                with open(file.name, "rb") as f:
                    file_hash = self._generate_hash(f.read())

                cache_path = self.cache_dir / f"{file_hash}.pkl"

                if self._is_cache_valid(cache_path):
                    logger.info(f"Loading from cache: {file.name}")
                    chunks = self._load_from_cache(cache_path)
                else:
                    logger.info(f"Processing and caching: {file.name}")
                    chunks = self._process_file(file)

                    if not chunks:
                        logger.warning(f"No content extracted from {file.name}")
                        continue

                    self._save_to_cache(chunks, cache_path)

                # Deduplicate chunks across files
                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode())
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)

            except Exception as e:
                logger.error(f"Failed to process {file.name}: {str(e)}")
                continue

        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    def _process_file(self, file) -> List:
        """
        Streamlit-cloud-safe document processing.
        Uses pdfplumber for PDFs and plain text loading for txt/md.
        """

        # ---------- PDF ----------
        if file.name.endswith(".pdf"):
            documents = []
            with pdfplumber.open(file.name) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": file.name,
                                    "page": page_idx + 1
                                }
                            )
                        )
            return documents

        # ---------- TEXT / MARKDOWN ----------
        elif file.name.endswith((".txt", ".md")):
            with open(file.name, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            return [
                Document(
                    page_content=text,
                    metadata={"source": file.name}
                )
            ]

        # ---------- UNSUPPORTED ----------
        else:
            logger.warning(f"Skipping unsupported file type: {file.name}")
            return []

    def _generate_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _save_to_cache(self, chunks: List, cache_path: Path):
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "timestamp": datetime.now().timestamp(),
                    "chunks": chunks,
                },
                f,
            )

    def _load_from_cache(self, cache_path: Path) -> List:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]

    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False

        cache_age = datetime.now() - datetime.fromtimestamp(
            cache_path.stat().st_mtime
        )
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)
