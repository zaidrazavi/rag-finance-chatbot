"""
Response Agent — formats the raw reasoning output into a clean business response.
"""

import os

import google.generativeai as genai

from app.prompts import get_prompt


class ResponseAgent:
    """
    Takes the raw reasoning output and formats it into a clean,
    professional response suitable for business users.
    """

    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
        self.prompt_version = get_prompt("response")

    def format_response(self, raw_answer: str) -> str:
        """Format reasoning output into a polished response."""
        prompt = self.prompt_version.template.format(raw_answer=raw_answer)
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return raw_answer  # fallback: return raw if formatting fails
