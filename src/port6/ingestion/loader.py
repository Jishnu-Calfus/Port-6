import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader

from port6.config import settings

logger = logging.getLogger(__name__)

# Below this many extracted characters, treat the page as having no real text
# layer (e.g. a scanned image) rather than as a legitimately short page.
MIN_EXTRACTABLE_CHARS = 20


@dataclass
class PageText:
    page_number: int  # 1-indexed, matches what a human would cite
    text: str


def _clean(text: str) -> str:
    """Normalize whitespace but keep paragraph breaks (\n\n) intact — the
    chunker splits on them first, so collapsing everything to one line would
    remove the exact structure it relies on."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ocr_page(pdf_path: Path, page_number: int) -> str:
    images = convert_from_path(str(pdf_path), first_page=page_number, last_page=page_number)
    if not images:
        return ""
    return pytesseract.image_to_string(images[0], lang=settings.ocr_language)


def load_pdf(pdf_path: str | Path) -> list[PageText]:
    """Extract per-page text from a PDF, falling back to OCR on pages with
    no extractable text layer (scanned pages)."""
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []

    for i, page in enumerate(reader.pages):
        page_number = i + 1
        text = page.extract_text() or ""

        if len(text.strip()) < MIN_EXTRACTABLE_CHARS:
            logger.info(
                "Page %d of %s has no text layer, falling back to OCR",
                page_number,
                pdf_path.name,
            )
            text = _ocr_page(pdf_path, page_number)

        pages.append(PageText(page_number=page_number, text=_clean(text)))

    return pages
