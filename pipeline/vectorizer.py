from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path

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
