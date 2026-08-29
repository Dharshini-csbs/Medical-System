"""Entry point for the Medical Management System."""

import os

from medical_management_system.chatbot.gemini_chatbot import GeminiChatbot
from medical_management_system.email_module.report_emailer import generate_report, send_report_email
from medical_management_system.ocr.ocr_processor import OCRProcessor
from medical_management_system.rag.faiss_rag import FaissRAG
from medical_management_system.tts.hf_tts import HuggingFaceTTS


def build_demo_rag() -> FaissRAG:
    rag = FaissRAG()
    rag.build_index(
        [
            "Hypertension can be managed with reduced sodium intake and regular exercise.",
            "Persistent chest pain should be evaluated immediately by a medical professional.",
            "For mild dehydration, increase oral fluid intake and monitor symptoms.",
        ]
    )
    return rag


def run_demo() -> None:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("Set GEMINI_API_KEY in your environment before running the demo.")

    rag = build_demo_rag()
    chatbot = GeminiChatbot(api_key=gemini_key)
    tts = HuggingFaceTTS()
    ocr = OCRProcessor()

    print("Medical Management System demo")
    question = input("Enter your medical question: ").strip()
    context_chunks = rag.retrieve(question, top_k=2)
    context = "\n".join(chunk["text"] for chunk in context_chunks)
    answer = chatbot.ask(question, context=context)
    print("\nChatbot answer:\n", answer)

    output_wav = tts.text_to_speech(answer, "response.wav")
    print(f"Audio saved to {output_wav}")

    image_path = input("Optional: provide document image path for OCR (or press Enter to skip): ").strip()
    ocr_text = ""
    if image_path:
        ocr_text = ocr.extract_text(image_path)
        print("\nOCR extracted text:\n", ocr_text)

    report = generate_report(
        patient_name=input("Patient name for report: ").strip() or "Unknown",
        findings=[answer] + ([ocr_text] if ocr_text else []),
        recommendations=["Consult a qualified physician for final diagnosis."],
    )
    print("\nGenerated report:\n")
    print(report)

    if input("Send this report by email? (y/N): ").strip().lower() == "y":
        send_report_email(
            smtp_host=os.getenv("SMTP_HOST", ""),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            sender_email=os.getenv("SMTP_SENDER_EMAIL", ""),
            sender_password=os.getenv("SMTP_SENDER_PASSWORD", ""),
            recipient_email=input("Recipient email: ").strip(),
            subject="Medical Report",
            report_text=report,
            use_tls=os.getenv("SMTP_USE_TLS", "true").lower() == "true",
        )
        print("Email sent.")


if __name__ == "__main__":
    run_demo()

