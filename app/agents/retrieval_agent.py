"""
Retrieval Agent — expands queries and fetches relevant context from vector store.
"""

import os
from typing import List, Tuple

import google.generativeai as genai
from langchain.schema import Document

from app.embeddings import semantic_search
from app.prompts import get_prompt


class RetrievalAgent:
    """
    Expands the raw user query using an LLM for better semantic recall,
    then retrieves top-k relevant chunks from the FAISS vector store.
    """

    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
        self.prompt_version = get_prompt("retrieval")

    def expand_query(self, raw_query: str) -> str:
        """Use Gemini to rewrite the query for better vector search recall."""
        prompt = self.prompt_version.template.format(query=raw_query)
        try:
            response = self.model.generate_content(prompt)
            expanded = response.text.strip()
            return expanded if expanded else raw_query
        except Exception:
            return raw_query  # fallback to original on API error

    def retrieve(self, query: str) -> Tuple[List[Document], List[float], str]:
        """
        Returns:
            docs       — retrieved LangChain Documents
            scores     — similarity scores (lower = more similar in FAISS L2)
            expanded_q — the rewritten query used for retrieval
        """
        expanded_query = self.expand_query(query)
        results = semantic_search(expanded_query)
        docs = [r[0] for r in results]
        scores = [float(r[1]) for r in results]
        return docs, scores, expanded_query
