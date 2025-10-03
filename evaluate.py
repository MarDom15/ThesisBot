from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# ------------------ CONFIG ------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------ MÉTRIQUES ------------------
def precision_recall_at_k(predicted, relevant, k=5):
    """
    predicted : liste des indices des passages récupérés par l'agent est ceci 
    relevant : liste des indices des passages réellement pertinents
    """
    pred_topk = set(predicted[:k])
    relevant_set = set(relevant)
    tp = len(pred_topk & relevant_set)
    precision = tp / len(pred_topk) if pred_topk else 0
    recall = tp / len(relevant_set) if relevant_set else 0
    return precision, recall

def coverage_score(synth_text, original_paragraphs):
    """
    Mesure la similarité moyenne entre la synthèse et les paragraphes originaux
    """
    embeddings = model.encode([synth_text] + original_paragraphs)
    synth_emb = embeddings[0]
    paras_emb = embeddings[1:]
    sim = cosine_similarity([synth_emb], paras_emb)
    return float(np.mean(sim))

# ------------------ TEST SIMPLE ------------------
if __name__ == "__main__":
    # Charger quelques paragraphes pour test
    with open("data/text_chunks.pkl", "rb") as f:
        paragraphs = pickle.load(f)

    # Exemple simple : les 5 premiers comme pertinents
    relevant_idx = list(range(5))
    predicted_idx = list(range(3, 8))

    p, r = precision_recall_at_k(predicted_idx, relevant_idx, k=5)
    print(f"Precision@5: {p:.2f}, Recall@5: {r:.2f}")

    # Coverage test
    sample_synth = "This is a test synthesis of the first five paragraphs."
    coverage = coverage_score(sample_synth, paragraphs[:5])
    print(f"Coverage score: {coverage:.2f}")
