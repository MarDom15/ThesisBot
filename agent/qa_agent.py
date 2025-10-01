import os
import pickle
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# ------------------ CONFIG ------------------
os.environ["HF_API_KEY"] = "YOUR_HF_API_KEY"   # <-- Mets ton vrai token Hugging Face
MODEL_NAME = "tiiuae/falcon-rw-1b"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Hugging Face pipeline (génération texte avec Falcon)
generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device=-1   # CPU (-1) ou GPU (0 si tu as CUDA dispo)
)

# ------------------ CHARGEMENT ------------------
with open("data/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

index = faiss.read_index("data/index.faiss")

with open("data/text_chunks.pkl", "rb") as f:
    data = pickle.load(f)
    paragraphs = data["paragraphs"]
    sources = data["sources"]

# ------------------ FONCTIONS ------------------
def retrieve_top_paragraphs(question, paragraphs, embeddings, index, top_k=5, subject=None):
    q_text = f"{question}"
    if subject:
        q_text = f"Sujet de thèse : {subject}\nQuestion : {question}"
    q_emb = embed_model.encode([q_text])
    D, I = index.search(q_emb, top_k)
    return [paragraphs[int(i)] for i in I[0] if int(i) < len(paragraphs)]

def reformulate_text(text, style="académique"):
    prompt = f"Reformule ce texte pour qu'il soit utilisable dans une thèse avec style {style} et sans plagiat :\n\n{text}"
    result = generator(
        prompt,
        max_new_tokens=200,        # combien de tokens on veut générer
        truncation=True,           # coupe l’entrée si trop longue
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

        top_paras = retrieve_top_paragraphs(question, paragraphs, embeddings, index, top_k=5, subject=subject)
        print("\n--- Top passages trouvés ---")
        for i, para in enumerate(top_paras, 1):
            print(f"[{i}] {para}\n")

        print("\n--- Reformulations ---")
        for i, para in enumerate(top_paras, 1):
            reformulated = reformulate_text(para)
            print(f"[{i}] {reformulated}\n")

