# Medical Management System

Python-based Medical Management System with modular components for:

- Gemini API chatbot
- RAG pipeline using FAISS vector database
- OCR extraction from medical document images
- Hugging Face Text-to-Speech output
- Medical report generation and email sending via `smtplib` and `email.message`

## Project structure

```text
medical_management_system/
  chatbot/gemini_chatbot.py
  rag/faiss_rag.py
  ocr/ocr_processor.py
  tts/hf_tts.py
  email_module/report_emailer.py
main.py
requirements.txt
.env.example
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy environment file and fill values:
   ```bash
   cp .env.example .env
   ```
3. Ensure Tesseract OCR is installed on your OS (required by `pytesseract`).

## Run

```bash
python main.py
```

The CLI demo retrieves relevant medical context via FAISS, asks Gemini, can run OCR on an uploaded image path, converts the response to speech, generates a report, and optionally emails it.
