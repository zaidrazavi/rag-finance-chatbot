"""
FAISS vector store using sentence-transformers (local, no API needed for embeddings).
"""

import os
from pathlib import Path
from typing import List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


VECTOR_STORE_PATH = Path("data/vectorstore")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K_RESULTS", 4))


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def chunk_documents(texts: List[str], metadatas: List[dict] = None) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    docs = []
    for i, text in enumerate(texts):
        meta = metadatas[i] if metadatas else {"source": f"doc_{i}"}
        chunks = splitter.create_documents([text], metadatas=[meta])
        docs.extend(chunks)
    return docs


def build_vectorstore(documents: List[Document]) -> FAISS:
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTOR_STORE_PATH))
    return vectorstore


def load_vectorstore() -> FAISS:
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError("No vector store found. Upload documents first.")
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def add_to_vectorstore(new_documents: List[Document]) -> FAISS:
    try:
        vectorstore = load_vectorstore()
        vectorstore.add_documents(new_documents)
    except FileNotFoundError:
        vectorstore = build_vectorstore(new_documents)
    vectorstore.save_local(str(VECTOR_STORE_PATH))
    return vectorstore


def semantic_search(query: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results
