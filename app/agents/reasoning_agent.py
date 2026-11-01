"""
Reasoning Agent — synthesizes retrieved context into a grounded answer.
"""

import os
from typing import List

import google.generativeai as genai
from langchain.schema import Document

from app.prompts import get_prompt


class ReasoningAgent:
    """
    Takes retrieved context chunks and the original query,
    then generates a grounded, step-by-step financial answer using Gemini.
    """

    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
        self.prompt_version = get_prompt("reasoning")

    def _format_context(self, docs: List[Document]) -> str:
        """Combine retrieved chunks into a single context block."""
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", f"chunk_{i}")
            parts.append(f"[Source {i} — {source}]\n{doc.page_content}")
        return "\n\n".join(parts)

    def reason(self, query: str, docs: List[Document]) -> str:
        """Generate a grounded answer from context."""
        if not docs:
            return "No relevant context found. Please upload relevant financial documents first."

        context = self._format_context(docs)
        prompt = self.prompt_version.template.format(context=context, query=query)

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Reasoning error: {str(e)}"
