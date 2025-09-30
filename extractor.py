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

def load_articles(folder=r"C:\Users\marti\Desktop\ThesisBot\data"):
    """
    Charge tous les PDF et TXT du dossier donné, retourne un dictionnaire :
    {nom_fichier: [liste des segments de texte]}
    """
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

if __name__ == "__main__":
    articles = load_articles()
    print(f"{len(articles)} articles chargés.")
    # Affiche les 2 premiers segments du premier article
    first_key = list(articles.keys())[0]
    print(f"\nPremiers segments de {first_key}:")
    for seg in articles[first_key][:2]:
        print(seg, "\n")
