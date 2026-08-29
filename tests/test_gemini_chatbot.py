import os
import unittest
from unittest.mock import Mock

from medical_management_system.chatbot.gemini_chatbot import GeminiChatbot


class GeminiChatbotTests(unittest.TestCase):
    def test_requires_api_key(self):
        previous = os.environ.pop("GEMINI_API_KEY", None)
        try:
            with self.assertRaises(ValueError):
                GeminiChatbot()
        finally:
            if previous is not None:
                os.environ["GEMINI_API_KEY"] = previous

    def test_ask_passes_context_to_model(self):
        bot = GeminiChatbot(api_key="test-key")
        model = Mock()
        model.generate_content.return_value.text = "ok"
        bot._model = model

        response = bot.ask("What is fever?", context="Fever is body temperature > 38C.")
        self.assertEqual("ok", response)
        prompt = model.generate_content.call_args.args[0]
        self.assertIn("Medical context", prompt)
        self.assertIn("What is fever?", prompt)


if __name__ == "__main__":
    unittest.main()

