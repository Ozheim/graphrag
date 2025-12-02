# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""PDF input loader for GraphRAG."""

import logging
from io import BytesIO
from pathlib import Path

import pandas as pd

from graphrag.config.models.input_config import InputConfig
from graphrag.index.input.util import load_files
from graphrag.index.utils.hashing import gen_sha512_hash
from graphrag.storage.pipeline_storage import PipelineStorage

logger = logging.getLogger(__name__)


async def load_pdf(
    config: InputConfig,
    storage: PipelineStorage,
) -> pd.DataFrame:
    """Load PDF inputs from a directory and extract text."""
    logger.info("Loading PDF files from %s", config.storage.base_dir)

    async def load_file(path: str, group: dict | None = None) -> pd.DataFrame:
        if group is None:
            group = {}

        # Read PDF file as bytes
        pdf_bytes = await storage.get(path, as_bytes=True)
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_bytes, path)
        
        new_item = {**group, "text": text}
        new_item["id"] = gen_sha512_hash(new_item, new_item.keys())
        new_item["title"] = str(Path(path).name)
        new_item["creation_date"] = await storage.get_creation_date(path)
        
        return pd.DataFrame([new_item])

    return await load_files(load_file, config, storage)


def extract_text_from_pdf(pdf_bytes: bytes, path: str) -> str:
    """
    Extract text from PDF bytes using available PDF library.
    
    Tries multiple libraries in order of preference:
    1. pdfplumber (best for complex PDFs with tables)
    2. pypdf (formerly PyPDF2, good for simple PDFs)
    3. pymupdf (fitz, very fast)
    
    Args:
        pdf_bytes: PDF file content as bytes
        path: File path for logging
        
    Returns:
        Extracted text from the PDF
    """
    text = ""
    
    # Try pdfplumber first (best for tables and complex layouts)
    try:
        import pdfplumber
        
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        if text.strip():
            logger.info("Extracted text from %s using pdfplumber", path)
            return text.strip()
    except ImportError:
        logger.debug("pdfplumber not available, trying pypdf")
    except Exception as e:
        logger.warning("Error extracting text with pdfplumber from %s: %s", path, e)
    
    # Try pypdf (PyPDF2 replacement)
    try:
        from pypdf import PdfReader
        
        pdf = PdfReader(BytesIO(pdf_bytes))
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        if text.strip():
            logger.info("Extracted text from %s using pypdf", path)
            return text.strip()
    except ImportError:
        logger.debug("pypdf not available, trying pymupdf")
    except Exception as e:
        logger.warning("Error extracting text with pypdf from %s: %s", path, e)
    
    # Try pymupdf (fitz) as fallback
    try:
        import fitz  # pymupdf
        
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in pdf:
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n\n"
        pdf.close()
        
        if text.strip():
            logger.info("Extracted text from %s using pymupdf", path)
            return text.strip()
    except ImportError:
        logger.debug("pymupdf not available")
    except Exception as e:
        logger.warning("Error extracting text with pymupdf from %s: %s", path, e)
    
    # If all methods failed
    if not text.strip():
        msg = (
            f"Could not extract text from PDF: {path}. "
            "Please install one of: pdfplumber, pypdf, or pymupdf. "
            "Install with: pip install pdfplumber"
        )
        raise ValueError(msg)
    
    return text.strip()

