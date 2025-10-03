# ui/app.py
import streamlit as st
from pathlib import Path

from pipeline.extractor import extract_from_pdf
from pipeline.vectorizer import embed_paragraphs, build_faiss_index, load_embeddings, load_faiss_index
from agent.qa_agent import retrieve_top_paragraphs
from agent.summarizer import synthesize_paragraphs
from evaluation.evaluate import coverage_score
from utils.file_utils import export_pdf

# ----------------------- Streamlit -----------------------
st.set_page_config(page_title="ThesisBot", layout="wide")
st.title("ThesisBot - Résumé automatique de PDF")

# Upload PDF
uploaded_file = st.file_uploader("Téléversez un fichier PDF", type=["pdf"])
TOP_K = st.number_input("Nombre de paragraphes pertinents à récupérer", min_value=1, max_value=20, value=5)

if uploaded_file is not None:
    st.success("PDF chargé avec succès !")
    
    # Extraction des paragraphes
    paragraphs = extract_from_pdf(uploaded_file)
    st.write(f"{len(paragraphs)} phrases extraites du PDF.")
    
    st.subheader("Aperçu des paragraphes extraits")
    for i, p in enumerate(paragraphs[:5], 1):
        st.write(f"{i}. {p[:300]}{'...' if len(p)>300 else ''}")
    
    # Embeddings + FAISS
    EMBEDDINGS_FILE = "data/embeddings.pkl"
    FAISS_INDEX_FILE = "data/faiss.index"
    
    if Path(EMBEDDINGS_FILE).exists() and Path(FAISS_INDEX_FILE).exists():
        st.info("Chargement des embeddings et de l'index FAISS existants...")
        embeddings = load_embeddings(EMBEDDINGS_FILE)
        faiss_index = load_faiss_index(FAISS_INDEX_FILE)
    else:
        st.info("Création des embeddings et de l'index FAISS...")
        embeddings = embed_paragraphs(paragraphs)
        faiss_index = build_faiss_index(embeddings)
        st.success("Embeddings et index FAISS créés.")
    
    # Question de l'utilisateur
    query = st.text_input("Entrez votre question :", "")
    
    if query and st.button("Récupérer les paragraphes pertinents et synthèse"):
        st.info("Recherche des paragraphes pertinents...")
        top_paragraphs = retrieve_top_paragraphs(query, paragraphs, embeddings, faiss_index, top_k=TOP_K)
        
        st.subheader(f"{len(top_paragraphs)} paragraphes pertinents :")
        for i, p in enumerate(top_paragraphs, 1):
            st.write(f"{i}. {p[:300]}{'...' if len(p)>300 else ''}")
        
        st.info("Synthèse des paragraphes...")
        summary = synthesize_paragraphs(top_paragraphs)
        st.subheader("Résumé généré")
        st.write(summary)
        
        # Coverage
        coverage = coverage_score(summary, top_paragraphs)
        st.write(f"Coverage score : {coverage:.2f}")
        
        # Export PDF
        export_path = "output/summary.pdf"
        export_pdf(summary, export_path)
        st.success(f"Synthèse exportée : {export_path}")
