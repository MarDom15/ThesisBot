import os
import pickle
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline

# ------------------ CONFIG ------------------
os.environ["HF_API_KEY"] = "YOUR_HF_API_KEY"   # Mets ta clé Hugging Face si nécessaire
MODEL_NAME = "tiiuae/falcon-rw-1b"

# Téléchargement des tokenizers NLTK (pour segmentation en phrases)
nltk.download("punkt")
nltk.download("punkt_tab")

# Pipeline HuggingFace
generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device=-1  # CPU ; mets device=0 si tu as un GPU
)

# Dossier data
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

text_chunks_path = os.path.join(data_dir, "text_chunks.pkl")
article_map_path = os.path.join(data_dir, "article_map.pkl")
summary_file = os.path.join(data_dir, "summary.txt")

# ------------------ EXTRACTION PDF ------------------
def extract_from_pdf(pdf_path):
    """Extrait le texte d'un PDF et le découpe en phrases."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # Découpe en phrases
    sentences = sent_tokenize(text)
    return sentences

# ------------------ NETTOYAGE ------------------
def clean_paragraphs(paragraphs):
    """Nettoie les segments extraits pour supprimer le bruit."""
    cleaned = []
    for p in paragraphs:
        p = p.strip()

        # Supprime si trop court
        if len(p) < 25:
            continue

        # Supprime les DOI / URLs
        if p.lower().startswith("doi") or "http" in p:
            continue

        # Supprime les lignes en majuscules (souvent titres, bruit)
        if p.isupper():
            continue

        # Supprime si plus de chiffres que de lettres
        letters = sum(c.isalpha() for c in p)
        digits = sum(c.isdigit() for c in p)
        if digits > letters:
            continue

        cleaned.append(p)
    return cleaned

# ------------------ SAUVEGARDE DES DONNÉES ------------------
def prepare_data(pdf_path):
    """Crée text_chunks.pkl et article_map.pkl à partir d'un PDF."""
    print(f"[DEBUG] Extraction depuis {pdf_path}...")
    paragraphs = extract_from_pdf(pdf_path)
    paragraphs = clean_paragraphs(paragraphs)

    # Sauvegarde des segments
    with open(text_chunks_path, "wb") as f:
        pickle.dump(paragraphs, f)
    print(f"Nombre total de segments après nettoyage : {len(paragraphs)}")
    print("Fichier text_chunks.pkl créé avec succès !")

    # Associer chaque segment à la source (ici le nom du PDF)
    pdf_name = os.path.basename(pdf_path)
    sources = [pdf_name for _ in range(len(paragraphs))]
    with open(article_map_path, "wb") as f:
        pickle.dump(sources, f)
    print("Fichier article_map.pkl créé avec succès !")

    return paragraphs, sources

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
        max_new_tokens=300,
        truncation=True,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return result[0]["generated_text"]

# ------------------ SYNTHÈSE EN BATCH ------------------
def batch_summarize(paragraphs, sources, batch_size=50):
    """Divise les segments en lots pour éviter de saturer le modèle."""
    summaries = []
    for i in range(0, len(paragraphs), batch_size):
        chunk_paras = paragraphs[i:i+batch_size]
        chunk_sources = sources[i:i+batch_size]
        summary = synthesize_paragraphs(chunk_paras, chunk_sources)
        summaries.append(summary)
        print(f"[DEBUG] Batch {i//batch_size+1} synthétisé ({len(chunk_paras)} segments)")

    # Fusion des résumés partiels
    final_summary = synthesize_paragraphs(
        summaries, ["Synthèse partielle"] * len(summaries)
    )
    return final_summary

# ------------------ BOUCLE PRINCIPALE ------------------
if __name__ == "__main__":
    pdf_file = os.path.join(data_dir, "s42484-022-00083-z.pdf")

    if not os.path.exists(text_chunks_path) or not os.path.exists(article_map_path):
        paragraphs, sources = prepare_data(pdf_file)
    else:
        with open(text_chunks_path, "rb") as f:
            paragraphs = pickle.load(f)
        with open(article_map_path, "rb") as f:
            sources = pickle.load(f)
        print("✅ Fichiers déjà existants, chargés depuis data/")

    print(f"{len(paragraphs)} segments disponibles pour la synthèse.")

    # ⚡ Synthèse par batch
    summary = batch_summarize(paragraphs, sources, batch_size=50)

    print("\n--- Synthèse finale ---\n")
    print(summary)

    # Sauvegarde dans un fichier
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\n✅ Synthèse finale sauvegardée dans {summary_file}")
