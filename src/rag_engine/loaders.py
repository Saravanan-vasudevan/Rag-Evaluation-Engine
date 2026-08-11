"""File readers for supported upload types (PDF, plain text, markdown)."""

import os
import tempfile

from pypdf import PdfReader


def load_pdf(file_bytes: bytes) -> str:
    # pypdf needs a real file path, so spool the upload to disk and clean up after
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        reader = PdfReader(tmp_path)
        pages = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n\n".join(pages)
    finally:
        os.unlink(tmp_path)


def load_text(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # older exports / Windows-authored files sometimes come through as latin-1
        return file_bytes.decode("latin-1")
