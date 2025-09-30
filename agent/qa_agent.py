from openai import OpenAI
from sentence_transformers import SentenceTransformer

client = OpenAI(api_key="TON_OPENAI_API_KEY")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

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
