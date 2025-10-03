import os
import pickle
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# ------------------ CONFIG ------------------
os.environ["HF_API_KEY"] = "YOUR_HF_API_KEY"   # <-- Replace with your actual Hugging Face key
MODEL_NAME = "tiiuae/falcon-rw-1b"

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Hugging Face text-generation pipeline (Falcon)
generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device=-1,   # CPU (-1) or GPU (0 if CUDA available)
    max_length=512
)

# ------------------ LOAD DATA ------------------
# Embeddings and FAISS index
with open("data/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

index = faiss.read_index("data/index.faiss")

# Load paragraphs
with open("data/text_chunks.pkl", "rb") as f:
    paragraphs = pickle.load(f)  
    sources = ["unknown"] * len(paragraphs)  # fallback if no sources available

# ------------------ FUNCTIONS ------------------
def retrieve_top_paragraphs(question, paragraphs, embeddings, index, top_k=10, subject=None):
    """
    Retrieve the top-k most relevant paragraphs from FAISS.
    """
    q_text = f"{question}"
    if subject:
        q_text = f"Thesis topic: {subject}\nQuestion: {question}"
    
    q_emb = embed_model.encode([q_text])
    D, I = index.search(q_emb, top_k)
    
    # Get valid indices
    results = [paragraphs[int(i)] for i in I[0] if int(i) < len(paragraphs)]
    if not results:
        results = ["No relevant passages found in the documents."]
    
    # Debug info
    print(f"[DEBUG] Indices found: {I.tolist()}")
    print(f"[DEBUG] Distances: {D.tolist()}")
    
    return results

def reformulate_text(text, style="academic"):
    """
    Reformulate a paragraph in an academic style without plagiarism.
    """
    prompt = f"Rephrase this text so it can be used in a thesis with {style} style and no plagiarism:\n\n{text}"
    result = generator(
        prompt,
        max_new_tokens=200,
        truncation=True,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return result[0]["generated_text"]

# ------------------ INTERACTIVE LOOP ------------------
if __name__ == "__main__":
    subject = input("Enter your thesis topic (optional): ").strip()
    
    while True:
        question = input("\nAsk your question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        top_paras = retrieve_top_paragraphs(question, paragraphs, embeddings, index, top_k=10, subject=subject)
        
        print("\n--- Top retrieved paragraphs ---")
        for i, para in enumerate(top_paras, 1):
            print(f"[{i}] {para}\n")
        
        print("\n--- Reformulated paragraphs ---")
        for i, para in enumerate(top_paras, 1):
            reformulated = reformulate_text(para)
            print(f"[{i}] {reformulated}\n")
