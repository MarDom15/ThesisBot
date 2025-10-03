from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path
import os

# Import the existing PDF extraction function
from extractor import extract_from_pdf  

# ------------------ CONFIG ------------------
MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_FOLDER = r"C:\Users\marti\Desktop\ThesisBot\data"

# ------------------ EMBEDDINGS ------------------
model = SentenceTransformer(MODEL_NAME)

def embed_paragraphs(paragraphs, save_path=None):
    """Compute embeddings for a list of paragraphs."""
    embeddings = model.encode(paragraphs, convert_to_numpy=True)
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(embeddings, f)
    return embeddings

def load_embeddings(path):
    """Load embeddings from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

# ------------------ FAISS INDEX ------------------
def build_faiss_index(embeddings, save_path=None):
    """Build a FAISS index from embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance
    index.add(embeddings)
    if save_path:
        faiss.write_index(index, str(save_path))
    return index

def load_faiss_index(path):
    """Load a FAISS index from disk."""
    return faiss.read_index(str(path))

# ------------------ LOAD PDF ARTICLES ------------------
def load_articles(folder_path):
    """Load all PDF files from a folder and extract sentences."""
    articles = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            sentences = extract_from_pdf(pdf_path)
            articles[filename] = sentences
    return articles

# ------------------ MAIN EXECUTION ------------------
if __name__ == "__main__":
    # 1️⃣ Load all paragraphs and map them to sources
    articles = load_articles(DATA_FOLDER)
    paragraphs = []
    article_map = []  # Track which PDF each segment comes from
    for name, segs in articles.items():
        for seg in segs:
            paragraphs.append(seg)
            article_map.append(name)

    print(f"Total segments: {len(paragraphs)}")

    # 2️⃣ Compute embeddings
    embeddings = embed_paragraphs(paragraphs, save_path="data/embeddings.pkl")
    print("Embeddings computed and saved.")

    # 3️⃣ Build FAISS index
    index = build_faiss_index(embeddings, save_path="data/index.faiss")
    print("FAISS index built and saved.")
