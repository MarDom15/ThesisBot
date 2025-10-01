import fitz  # PyMuPDF pour PDF
from pathlib import Path
import re
import pickle

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    # Debug : affiche un extrait pour vérifier
    print(f"[DEBUG] {file_path} extrait :\n{text[:500]}...\n")
    return text

def segment_text(text, min_length=20):
    # Segmentation par paragraphes (lignes vides)
    segments = re.split(r'\n\n+', text)
    clean_segments = [seg.strip() for seg in segments if len(seg.strip()) > min_length]
    return clean_segments

def load_articles(folder=r"C:\Users\marti\Desktop\ThesisBot\data"):
    """
    Charge tous les PDF et TXT du dossier donné, retourne un dictionnaire :
    {nom_fichier: [liste des segments de texte]}
    """
    articles = {}
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"[ERREUR] Dossier {folder} introuvable !")
        return articles

    for file in folder_path.glob("*"):
        if file.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file)
            articles[file.stem] = segment_text(text)
        elif file.suffix.lower() == ".txt":
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
            articles[file.stem] = segment_text(text)

    return articles

if __name__ == "__main__":
    articles = load_articles()  # utilise directement ton chemin absolu
    paragraphs = []
    sources = []

    for title, paras in articles.items():
        for p in paras:
            if len(p.strip()) > 0:
                paragraphs.append(p)
                sources.append(title)

    if not paragraphs:
        print("[ATTENTION] Aucun texte extrait. Vérifie que tes PDFs contiennent du texte sélectionnable.")
    else:
        with open(r"C:\Users\marti\Desktop\ThesisBot\data\text_chunks.pkl", "wb") as f:
            pickle.dump({"paragraphs": paragraphs, "sources": sources}, f)

        print(f"Nombre total de segments : {len(paragraphs)}")
        print("Fichier text_chunks.pkl créé avec succès !")
