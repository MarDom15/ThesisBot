from openai import OpenAI
from sentence_transformers import SentenceTransformer
import pickle
import faiss

# ------------------ CONFIG ------------------
client = OpenAI(api_key="TON_OPENAI_API_KEY")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------ CHARGEMENT ------------------
with open("data/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

index = faiss.read_index("data/index.faiss")

with open("data/text_chunks.pkl", "rb") as f:
    paragraphs = pickle.load(f)  # liste de tous les segments

# ------------------ FONCTIONS ------------------
def retrieve_top_paragraphs(question, paragraphs, embeddings, index, top_k=5, subject=None):
    q_text = f"{question}"
    if subject:
        q_text = f"Sujet de thèse : {subject}\nQuestion : {question}"
    q_emb = embed_model.encode([q_text])
    D, I = index.search(q_emb, top_k)
    return [paragraphs[i] for i in I[0]]

def reformulate_text(text, style="académique"):
    prompt = f"Reformule ce texte pour qu'il soit utilisable dans une thèse avec style {style} et sans plagiat :\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content

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
