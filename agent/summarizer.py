from openai import OpenAI
import pickle

# ------------------ CONFIG ------------------
client = OpenAI(api_key="TON_OPENAI_API_KEY")

# ------------------ CHARGEMENT DES PARAGRAPHES ------------------
with open("data/text_chunks.pkl", "rb") as f:
    paragraphs = pickle.load(f)  # liste des segments
with open("data/article_map.pkl", "rb") as f:
    sources = pickle.load(f)     # liste des sources correspondant à chaque segment

# ------------------ FONCTION DE SYNTHESE ------------------
def synthesize_paragraphs(paragraphs, sources=None):
    combined = ""
    for i, p in enumerate(paragraphs):
        source = sources[i] if sources else "Source inconnue"
        combined += f"[{source}]: {p}\n"
    prompt = f"Fais une synthèse claire et concise de ces passages avec reformulation académique, en conservant les références source :\n\n{combined}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content

# ------------------ BOUCLE INTERACTIVE ------------------
if __name__ == "__main__":
    print(f"{len(paragraphs)} segments disponibles pour la synthèse.")
    # exemple simple : synthèse des 5 premiers segments
    sample_paras = paragraphs[:5]
    sample_sources = sources[:5] if sources else None

    summary = synthesize_paragraphs(sample_paras, sample_sources)
    print("\n--- Synthèse générée ---\n")
    print(summary)
