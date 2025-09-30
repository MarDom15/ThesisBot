import streamlit as st
from pathlib import Path
import fitz  # PyMuPDF
import pickle

# Import des modules internes
from pipeline.extractor import load_articles
from pipeline.vectorizer import embed_paragraphs, build_faiss_index, load_embeddings, load_faiss_index
from agent.qa_agent import retrieve_top_paragraphs, reformulate_text
from agent.summarizer import synthesize_paragraphs
from evaluation.evaluate import coverage_score
from utils.file_utils import export_pdf

st.set_page_config(page_title="📝 Thesis Assistant AI", layout="wide")
st.title("📝 Thesis Assistant AI Optimisé")

# ------------------ Upload d'articles ------------------
uploaded_files = st.file_uploader(
    "Téléversez vos articles (PDF/TXT)",
    accept_multiple_files=True
)

articles = {}
if uploaded_files:
    for file in uploaded_files:
        if file.name.lower().endswith(".pdf"):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = "".join([page.get_text() for page in doc])
            articles[file.name] = text.split("\n\n")
        elif file.name.lower().endswith(".txt"):
            text = file.read().decode("utf-8")
            articles[file.name] = text.split("\n\n")
else:
    articles = load_articles()

# ------------------ Préparation des paragraphes et sources ------------------
paragraphs = []
sources = []
for title, paras in articles.items():
    for p in paras:
        if len(p.strip()) > 20:
            paragraphs.append(p)
            sources.append(title)

# ------------------ Embeddings et FAISS ------------------
if Path("data/embeddings.pkl").exists() and Path("data/index.faiss").exists():
    embeddings = load_embeddings("data/embeddings.pkl")
    index = load_faiss_index("data/index.faiss")
else:
    embeddings = embed_paragraphs(paragraphs, save_path="data/embeddings.pkl")
    index = build_faiss_index(embeddings, save_path="data/index.faiss")

# ------------------ Inputs utilisateur ------------------
subject = st.text_input("Sujet de la thèse / Contexte :")
question = st.text_input("Pose ta question :")
top_k = st.slider("Nombre de passages top-k :", min_value=1, max_value=10, value=5)

if st.button("Obtenir réponse"):
    if not question.strip():
        st.warning("Veuillez entrer une question.")
    else:
        # Récupération des passages pertinents
        top_paras = retrieve_top_paragraphs(question, paragraphs, embeddings, index, top_k=top_k, subject=subject)
        
        st.subheader("Passages pertinents avec reformulation :")
        for i, p in enumerate(top_paras, 1):
            st.write(f"**Original [{sources[i-1]}] :** {p}")
            reformulated = reformulate_text(p)
            st.write(f"**Reformulé :** {reformulated}")
            st.write("---")
        
        # Synthèse globale
        st.subheader("Synthèse globale :")
        synth = synthesize_paragraphs(top_paras, sources[:len(top_paras)])
        st.write(synth)
        
        # Score de couverture
        st.subheader("Score de couverture :")
        score = coverage_score(synth, top_paras)
        st.write(f"{score:.2f}")
        
        # Export PDF
        if st.button("Exporter synthèse PDF"):
            export_pdf(synth, "synthese_these.pdf")
            st.success("PDF exporté !")
