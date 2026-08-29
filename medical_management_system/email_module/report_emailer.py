import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Iterable


def generate_report(patient_name: str, findings: Iterable[str], recommendations: Iterable[str]) -> str:
    findings_text = "\n".join(f"- {item}" for item in findings) or "- N/A"
    recommendations_text = "\n".join(f"- {item}" for item in recommendations) or "- N/A"
    return (
        f"Medical Report\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n"
        f"Patient: {patient_name}\n\n"
        f"Findings:\n{findings_text}\n\n"
        f"Recommendations:\n{recommendations_text}\n"
    )


def send_report_email(
    smtp_host: str,
    smtp_port: int,
    sender_email: str,
    sender_password: str,
    recipient_email: str,
    subject: str,
    report_text: str,
    use_tls: bool = True,
) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = recipient_email
    message.set_content(report_text)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
        if use_tls:
            smtp.starttls()
        smtp.login(sender_email, sender_password)
        smtp.send_message(message)
