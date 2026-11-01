"""
RAG Pipeline — orchestrates the full Retrieval → Reasoning → Response flow.
"""

import logging
from typing import List

from langchain.schema import Document

from app.agents.retrieval_agent import RetrievalAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.response_agent import ResponseAgent
from app.hallucination_check import HallucinationChecker
from app.embeddings import chunk_documents, add_to_vectorstore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Hierarchical 3-agent RAG pipeline:
      1. RetrievalAgent  — expands query + fetches relevant chunks
      2. ReasoningAgent  — generates grounded answer from context
      3. ResponseAgent   — formats answer for business users
      4. HallucinationChecker — scores answer grounding (self-reflection)
    """

    def __init__(self):
        self.retrieval_agent = RetrievalAgent()
        self.reasoning_agent = ReasoningAgent()
        self.response_agent = ResponseAgent()
        self.hallucination_checker = HallucinationChecker()

    def ingest(self, texts: List[str], metadatas: List[dict] = None) -> int:
        """Chunk and embed documents into the vector store. Returns chunk count."""
        documents = chunk_documents(texts, metadatas)
        add_to_vectorstore(documents)
        logger.info(f"Ingested {len(documents)} chunks into vector store.")
        return len(documents)

    def query(self, user_query: str) -> dict:
        """
        Full pipeline execution.

        Returns:
            answer          — final formatted response
            grounding_score — hallucination check score (0–1)
            grounding_passed— bool
            warning         — hallucination warning if applicable
            expanded_query  — rewritten query used for retrieval
            sources         — list of source metadata from retrieved chunks
            prompt_versions — active prompt versions used
        """
        logger.info(f"Query received: {user_query}")

        # Step 1 — Retrieve
        docs, scores, expanded_query = self.retrieval_agent.retrieve(user_query)
        logger.info(f"Retrieved {len(docs)} chunks. Expanded query: {expanded_query}")

        # Step 2 — Reason
        raw_answer = self.reasoning_agent.reason(user_query, docs)

        # Step 3 — Format Response
        final_answer = self.response_agent.format_response(raw_answer)

        # Step 4 — Hallucination Check
        context_text = "\n\n".join([d.page_content for d in docs])
        grounding = self.hallucination_checker.check(context_text, final_answer)

        # Build source list
        sources = []
        for doc, score in zip(docs, scores):
            sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "preview": doc.page_content[:120] + "...",
                "similarity_score": round(score, 4),
            })

        return {
            "answer": final_answer,
            "grounding_score": grounding["score"],
            "grounding_passed": grounding["passed"],
            "warning": grounding["warning"],
            "expanded_query": expanded_query,
            "sources": sources,
            "prompt_versions": {
                "retrieval": self.retrieval_agent.prompt_version.version,
                "reasoning": self.reasoning_agent.prompt_version.version,
                "response": self.response_agent.prompt_version.version,
            },
        }
