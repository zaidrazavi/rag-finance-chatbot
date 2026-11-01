"""
Versioned prompt templates for the RAG Finance Chatbot.
Each prompt version is tracked for A/B testing and performance QA.
"""

from dataclasses import dataclass


@dataclass
class PromptVersion:
    version: str
    description: str
    template: str


# ── Retrieval Agent Prompts ──────────────────────────────────────────────────

RETRIEVAL_PROMPT_V1 = PromptVersion(
    version="retrieval-v1.0",
    description="Basic query expansion for semantic search",
    template="""You are a retrieval specialist for financial documents.
Given the user query below, rewrite it into a precise search query that will
retrieve the most relevant financial context from a vector database.

User Query: {query}

Rewritten Search Query:"""
)

RETRIEVAL_PROMPT_V2 = PromptVersion(
    version="retrieval-v2.0",
    description="Multi-angle query expansion with finance domain focus",
    template="""You are a financial document retrieval specialist.
Expand the user query into 2-3 search angles to maximize recall from a
finance knowledge base (invoices, AR, payments, transactions).

User Query: {query}

Return only the expanded query, no explanation:"""
)

# ── Reasoning Agent Prompts ──────────────────────────────────────────────────

REASONING_PROMPT_V1 = PromptVersion(
    version="reasoning-v1.0",
    description="Standard finance QA with grounding instructions",
    template="""You are a financial analyst AI assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not found in the context, say "I don't have enough information to answer this."
Never make up financial figures, dates, or names.

Context:
{context}

Question: {query}

Answer:"""
)

REASONING_PROMPT_V2 = PromptVersion(
    version="reasoning-v2.0",
    description="Chain-of-thought reasoning with source citation",
    template="""You are a precise financial analyst AI assistant.
Use the context below to answer the question step by step.
Cite which part of the context supports your answer.
If the answer cannot be found in the context, explicitly say so — do not guess.

Rules:
- Only use facts from the provided context
- Always mention the source of key figures
- Keep the answer concise and professional
- Never fabricate numbers, dates, vendor names, or invoice details

Context:
{context}

Question: {query}

Step-by-step Answer:"""
)

# ── Response Agent Prompts ───────────────────────────────────────────────────

RESPONSE_PROMPT_V1 = PromptVersion(
    version="response-v1.0",
    description="Clean formatting of reasoning output",
    template="""Format the following financial analysis into a clear,
professional response for a business user. Be concise.

Raw Analysis: {raw_answer}

Formatted Response:"""
)

# ── Hallucination Check Prompts ──────────────────────────────────────────────

HALLUCINATION_PROMPT_V1 = PromptVersion(
    version="hallucination-v1.0",
    description="Self-reflection score for grounding verification",
    template="""You are a fact-checking AI. Score how well the answer is grounded
in the provided context on a scale of 0.0 to 1.0.

1.0 = Every claim in the answer is directly supported by the context
0.5 = Some claims are supported, some are inferred
0.0 = The answer contains fabricated information not in the context

Context: {context}
Answer: {answer}

Return ONLY a float between 0.0 and 1.0, nothing else:"""
)

# ── Active Prompt Registry ───────────────────────────────────────────────────

ACTIVE_PROMPTS = {
    "retrieval": RETRIEVAL_PROMPT_V2,
    "reasoning": REASONING_PROMPT_V2,
    "response": RESPONSE_PROMPT_V1,
    "hallucination": HALLUCINATION_PROMPT_V1,
}


def get_prompt(name: str) -> PromptVersion:
    if name not in ACTIVE_PROMPTS:
        raise ValueError(f"Unknown prompt: '{name}'. Available: {list(ACTIVE_PROMPTS.keys())}")
    return ACTIVE_PROMPTS[name]


def list_prompts() -> dict:
    return {k: {"version": v.version, "description": v.description}
            for k, v in ACTIVE_PROMPTS.items()}
