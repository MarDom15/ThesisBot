from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def precision_recall_at_k(predicted, relevant, k=5):
    pred_topk = set(predicted[:k])
    relevant_set = set(relevant)
    tp = len(pred_topk & relevant_set)
    precision = tp / len(pred_topk) if pred_topk else 0
    recall = tp / len(relevant_set) if relevant_set else 0
    return precision, recall

def coverage_score(synth_text, original_paragraphs):
    embeddings = model.encode([synth_text] + original_paragraphs)
    synth_emb = embeddings[0]
    paras_emb = embeddings[1:]
    sim = cosine_similarity([synth_emb], paras_emb)
    return float(np.mean(sim))
