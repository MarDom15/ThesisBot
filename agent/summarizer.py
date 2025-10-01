import pickle
from transformers import pipeline
import os

# ------------------ CONFIG ------------------
# Mets ta clé Hugging Face si besoin
os.environ["HF_API_KEY"] = "YOUR_HF_API_KEY"
MODEL_NAME = "tiiuae/falcon-rw-1b"

generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device=-1  # CPU ; mets device=0 si tu as un GPU
)

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

    prompt = (
        "Fais une synthèse claire et concise de ces passages avec reformulation académique, "
        "en conservant les références source :\n\n" + combined
    )

    result = generator(
        prompt,
        max_new_tokens=300,    # combien de tokens générer
        truncation=True,       # coupe si le texte est trop long
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return result[0]["generated_text"]

# ------------------ BOUCLE INTERACTIVE ------------------
if __name__ == "__main__":
    print(f"{len(paragraphs)} segments disponibles pour la synthèse.")
    # exemple simple : synthèse des 5 premiers segments
    sample_paras = paragraphs[:5]
    sample_sources = sources[:5] if sources else None

    summary = synthesize_paragraphs(sample_paras, sample_sources)
    print("\n--- Synthèse générée ---\n")
    print(summary)
