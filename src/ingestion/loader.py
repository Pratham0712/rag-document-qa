import os
from pypdf import PdfReader
from dataclasses import dataclass


@dataclass
class DocumentPage:
    page_number: int
    text: str
    source: str


def load_pdf(file_path: str) -> list[DocumentPage]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if not file_path.endswith(".pdf"):
        raise ValueError(f"File must be a PDF: {file_path}")

    reader = PdfReader(file_path)

    if len(reader.pages) == 0:
        raise ValueError(f"PDF has no pages: {file_path}")

    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if text and text.strip():
            pages.append(DocumentPage(
                page_number=i + 1,
                text=text.strip(),
                source=os.path.basename(file_path)
            ))

    print(f"Loaded {len(pages)} pages from {os.path.basename(file_path)}")
    return pages