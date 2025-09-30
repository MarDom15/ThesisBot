import fitz  # PyMuPDF pour PDF
from pathlib import Path
import re

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def segment_text(text):
    # Segmentation par paragraphes et titres (ligne vide ou majuscules)
    segments = re.split(r'\n\n+', text)
    clean_segments = [seg.strip() for seg in segments if len(seg.strip())>50]
    return clean_segments

def load_articles(folder="data/articles"):
    articles = {}
    for file in Path(folder).glob("*"):
        if file.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file)
            articles[file.stem] = segment_text(text)
        elif file.suffix.lower() == ".txt":
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
            articles[file.stem] = segment_text(text)
    return articles
