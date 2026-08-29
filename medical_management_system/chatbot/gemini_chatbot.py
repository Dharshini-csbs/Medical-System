import os
from typing import Optional


class GeminiChatbot:
    """Simple Gemini API wrapper for medical conversations."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash") -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required.")
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        return self._model

    def ask(self, question: str, context: Optional[str] = None) -> str:
        prompt = question if not context else f"Medical context:\n{context}\n\nUser question:\n{question}"
        response = self._get_model().generate_content(prompt)
        return (getattr(response, "text", "") or "").strip()

