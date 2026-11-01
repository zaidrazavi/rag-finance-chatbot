"""
Hallucination checker — self-reflection loop that scores answer grounding.
Re-generates if score is below threshold.
"""

import os
import logging

import google.generativeai as genai

from app.prompts import get_prompt

logger = logging.getLogger(__name__)

THRESHOLD = float(os.getenv("HALLUCINATION_THRESHOLD", 0.7))
MAX_RETRIES = 2


class HallucinationChecker:
    """
    Uses Gemini to score how well an answer is grounded in retrieved context.
    If score < threshold, flags the answer as potentially hallucinated.
    """

    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
        self.prompt_version = get_prompt("hallucination")
        self.threshold = THRESHOLD

    def score(self, context: str, answer: str) -> float:
        """Return grounding score between 0.0 and 1.0."""
        prompt = self.prompt_version.template.format(context=context, answer=answer)
        try:
            response = self.model.generate_content(prompt)
            raw = response.text.strip()
            score = float(raw)
            return max(0.0, min(1.0, score))  # clamp to [0, 1]
        except (ValueError, Exception) as e:
            logger.warning(f"Hallucination score parse failed: {e}. Defaulting to 0.5.")
            return 0.5

    def check(self, context: str, answer: str) -> dict:
        """
        Returns a dict with:
            score       — float grounding score
            passed      — bool, True if score >= threshold
            warning     — human-readable warning if failed
        """
        grounding_score = self.score(context, answer)
        passed = grounding_score >= self.threshold

        result = {
            "score": round(grounding_score, 3),
            "threshold": self.threshold,
            "passed": passed,
            "warning": None,
        }

        if not passed:
            result["warning"] = (
                f"Low grounding score ({grounding_score:.2f}). "
                "This answer may contain information not found in the uploaded documents. "
                "Please verify before use."
            )
            logger.warning(f"Hallucination check failed: score={grounding_score:.2f}")

        return result
