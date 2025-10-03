import sys
import os
from pathlib import Path

# Ajouter le dossier parent au path pour retrouver les modules à la racine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Modules internes
from pipeline.extractor import extract_from_pdf
from pipeline.vectorizer import embed_paragraphs, build_faiss_index, load_embeddings, load_faiss_index
from agent.qa_agent import retrieve_top_paragraphs, reformulate_text
from agent.summarizer import synthesize_paragraphs
from evaluation.evaluate import coverage_score
from utils.file_utils import export_pdf

# ---------------- CONFIG ----------------
PDF_PATH = "data/example.pdf"  # chemin vers ton PDF
TOP_K = 5  # nombre de paragraphes à récupérer
EMBEDDINGS_FILE = "data/embeddings.pkl"
FAISS_INDEX_FILE = "data/faiss.index"

def main():
    # 1. Extraction PDF
    print("[INFO] Extraction du PDF...")
    paragraphs = extract_from_pdf(PDF_PATH)
    print(f"[INFO] {len(paragraphs)} phrases extraites.")

    # 2. Embedding + FAISS
    if Path(EMBEDDINGS_FILE).exists() and Path(FAISS_INDEX_FILE).exists():
        print("[INFO] Chargement des embeddings et de l'index FAISS existants...")
        embeddings = load_embeddings(EMBEDDINGS_FILE)
        faiss_index = load_faiss_index(FAISS_INDEX_FILE)
    else:
        print("[INFO] Création des embeddings et de l'index FAISS...")
        embeddings = embed_paragraphs(paragraphs)
        faiss_index = build_faiss_index(embeddings)
        print("[INFO] Sauvegarde embeddings et FAISS...")
    
    # 3. Recherche QA
    query = input("Entrez votre question : ")
    top_paragraphs = retrieve_top_paragraphs(query, paragraphs, embeddings, faiss_index, top_k=TOP_K)
    
    print(f"[INFO] {len(top_paragraphs)} paragraphes pertinents récupérés :")
    for i, p in enumerate(top_paragraphs):
        print(f"{i+1}. {p[:100]}...")

    # 4. Synthèse
    print("[INFO] Synthèse des paragraphes...")
    summary = synthesize_paragraphs(top_paragraphs)
    print("\n--- SYNTHÈSE ---")
    print(summary)

    # 5. Évaluation coverage (optionnel)
    coverage = coverage_score(summary, top_paragraphs)
    print(f"\n[INFO] Coverage score : {coverage:.2f}")

    # 6. Export PDF (optionnel)
    export_path = "output/summary.pdf"
    export_pdf(summary, export_path)
    print(f"[INFO] Synthèse exportée dans {export_path}")

if __name__ == "__main__":
    main()
