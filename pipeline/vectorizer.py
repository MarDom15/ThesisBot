from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path
from pipeline.extractor import load_articles  # importe ta fonction d'extraction

# Modèle d'embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_paragraphs(paragraphs, save_path=None):
    embeddings = model.encode(paragraphs, convert_to_numpy=True)
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(embeddings, f)
    return embeddings

def load_embeddings(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def build_faiss_index(embeddings, save_path=None):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    if save_path:
        faiss.write_index(index, str(save_path))
    return index

def load_faiss_index(path):
    return faiss.read_index(str(path))

# --- Partie exécutable ---
if __name__ == "__main__":
    # 1️⃣ Charger tous les segments
    articles = load_articles(r"C:\Users\marti\Desktop\ThesisBot\data")
    paragraphs = []
    article_map = []  # Pour retrouver la source de chaque segment
    for name, segs in articles.items():
        for seg in segs:
            paragraphs.append(seg)
            article_map.append(name)

    print(f"Nombre total de segments : {len(paragraphs)}")

    # 2️⃣ Calcul des embeddings
    embeddings = embed_paragraphs(paragraphs, save_path="data/embeddings.pkl")
    print("Embeddings calculés et sauvegardés.")

    # 3️⃣ Construction de l'index FAISS
    index = build_faiss_index(embeddings, save_path="data/index.faiss")
    print("Index FAISS construit et sauvegardé.")
