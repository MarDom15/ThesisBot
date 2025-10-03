import os
import pickle
from openai import OpenAI

# ------------------ CONFIG ------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("La variable d'environnement OPENAI_API_KEY n'est pas définie.")
client = OpenAI(api_key=api_key)

# ------------------ CHARGEMENT DES PARAGRAPHES ------------------
try:
    with open("data/text_chunks.pkl", "rb") as f:
        paragraphs = pickle.load(f)  # liste des segments
except FileNotFoundError:
    raise FileNotFoundError("Le fichier 'data/text_chunks.pkl' est introuvable.")

try:
    with open("data/article_map.pkl", "rb") as f:
        sources = pickle.load(f)     # liste des sources correspondant à chaque segment
except FileNotFoundError:
    sources = None  # si tu n'as pas ce fichier, ça fonctionne quand même

# ------------------ FONCTION DE SYNTHESE ------------------
def synthesize_paragraphs(paragraphs, sources=None):
    combined = ""
    for i, p in enumerate(paragraphs):
        source = sources[i] if sources else "Source inconnue"
        combined += f"[{source}]: {p}\n"

    prompt = (
        "Fais une synthèse claire et concise de ces passages "
        "avec reformulation académique, en conservant les références source :\n\n"
        f"{combined}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

# ------------------ BOUCLE INTERACTIVE ------------------
if __name__ == "__main__":
    total_segments = len(paragraphs)
    print(f"{total_segments} segments disponibles pour la synthèse (dans data\\text_chunks.pkl).")
    
    while True:
        user_input = input(
            "Combien de segments à synthétiser ? (entrez 'all' pour tout synthétiser, 0 pour quitter) : "
        ).strip()

        if user_input == "0":
            print("Au revoir !")
            break
        elif user_input.lower() == "all":
            sample_paras = paragraphs
            sample_sources = sources
        else:
            try:
                n = int(user_input)
                if n > total_segments:
                    print(f"Il n'y a que {total_segments} segments. Je vais synthétiser tous.")
                    n = total_segments
                sample_paras = paragraphs[:n]
                sample_sources = sources[:n] if sources else None
            except ValueError:
                print("Entrée invalide. Tapez un nombre, 'all', ou 0 pour quitter.")
                continue

        print("\n--- Synthèse en cours ---\n")
        try:
            summary = synthesize_paragraphs(sample_paras, sample_sources)
            print("\n--- Synthèse générée ---\n")
            print(summary)
        except Exception as e:
            print(f"[ERREUR] Impossible de générer la synthèse : {e}")
