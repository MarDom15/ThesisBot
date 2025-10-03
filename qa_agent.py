import os
import pickle
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# ------------------ CONFIG ------------------
os.environ["HF_API_KEY"] = "YOUR_HF_API_KEY"   # <-- Mets ton vrai token Hugging Face
MODEL_NAME = "tiiuae/falcon-rw-1b"

# Modèle d'embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Hugging Face pipeline (génération texte avec Falcon)
generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device=-1,   # CPU (-1) ou GPU (0 si CUDA dispo)
    max_length=512
)

# ------------------ CHARGEMENT ------------------
# Embeddings et index FAISS
with open("data/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

index = faiss.read_index("data/index.faiss")

# Chargement des textes
with open("data/text_chunks.pkl", "rb") as f:
    paragraphs = pickle.load(f)  # liste de paragraphes
    sources = ["inconnu"] * len(paragraphs)  # si pas de sources précises

# ------------------ FONCTIONS ------------------
def retrieve_top_paragraphs(question, paragraphs, embeddings, index, top_k=10, subject=None):
    """
    Recherche les passages les plus pertinents dans l'index FAISS.
    top_k = nombre de passages à récupérer
    """
    q_text = f"{question}"
    if subject:
        q_text = f"Sujet de thèse : {subject}\nQuestion : {question}"
    
    q_emb = embed_model.encode([q_text])
    D, I = index.search(q_emb, top_k)
    
    # Filtrer les indices valides
    results = [paragraphs[int(i)] for i in I[0] if int(i) < len(paragraphs)]
    if not results:
        results = ["Aucun passage pertinent trouvé dans les documents."]
    
    # Debug
    print(f"[DEBUG] Indices trouvés : {I.tolist()}")
    print(f"[DEBUG] Distances : {D.tolist()}")
    
    return results

def reformulate_text(text, style="académique"):
    """
    Reformule un texte pour un style académique et sans plagiat.
    """
    prompt = f"Reformule ce texte pour qu'il soit utilisable dans une thèse avec style {style} et sans plagiat :\n\n{text}"
    result = generator(
        prompt,
        max_new_tokens=200,
        truncation=True,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return result[0]["generated_text"]

# ------------------ BOUCLE INTERACTIVE ------------------
if __name__ == "__main__":
    subject = input("Entrez le sujet de votre thèse (optionnel) : ").strip()
    
    while True:
        question = input("\nPosez votre question (ou tapez 'exit') : ")
        if question.lower() == "exit":
            break

        top_paras = retrieve_top_paragraphs(question, paragraphs, embeddings, index, top_k=10, subject=subject)
        
        print("\n--- Top passages trouvés ---")
        for i, para in enumerate(top_paras, 1):
            print(f"[{i}] {para}\n")
        
        print("\n--- Reformulations ---")
        for i, para in enumerate(top_paras, 1):
            reformulated = reformulate_text(para)
            print(f"[{i}] {reformulated}\n")
