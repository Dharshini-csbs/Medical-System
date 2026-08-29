class OCRProcessor:
    """Extract text from medical documents via Tesseract OCR."""

    def __init__(self, lang: str = "eng") -> None:
        self.lang = lang

    def extract_text(self, image_path: str) -> str:
        from PIL import Image
        import pytesseract

        with Image.open(image_path) as image:
            return pytesseract.image_to_string(image, lang=self.lang).strip()

