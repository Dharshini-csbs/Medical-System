import unittest
from unittest.mock import patch

from medical_management_system.email_module.report_emailer import generate_report, send_report_email


class ReportEmailerTests(unittest.TestCase):
    def test_generate_report_contains_expected_sections(self):
        report = generate_report("John Doe", ["Finding A"], ["Recommendation B"])
        self.assertIn("Patient: John Doe", report)
        self.assertIn("Findings:\n- Finding A", report)
        self.assertIn("Recommendations:\n- Recommendation B", report)

    @patch("medical_management_system.email_module.report_emailer.smtplib.SMTP")
    def test_send_report_email_uses_smtp(self, smtp_mock):
        send_report_email(
            smtp_host="smtp.example.com",
            smtp_port=587,
            sender_email="doctor@example.com",
            sender_password="password",
            recipient_email="patient@example.com",
            subject="Report",
            report_text="Body",
        )

        smtp_mock.assert_called_once_with("smtp.example.com", 587, timeout=30)
        smtp_instance = smtp_mock.return_value.__enter__.return_value
        smtp_instance.starttls.assert_called_once()
        smtp_instance.login.assert_called_once_with("doctor@example.com", "password")
        smtp_instance.send_message.assert_called_once()


if __name__ == "__main__":
    unittest.main()

