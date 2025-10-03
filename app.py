# ui/app_streamlit.py
import sys
import os
from pathlib import Path
import streamlit as st

# Add parent folder to sys.path to locate modules in the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Internal modules
from pipeline.extractor import extract_from_pdf
from pipeline.vectorizer import embed_paragraphs, build_faiss_index, load_embeddings, load_faiss_index
from agent.qa_agent import retrieve_top_paragraphs
from agent.summarizer import synthesize_paragraphs
from evaluation.evaluate import coverage_score
from utils.file_utils import export_pdf

# ---------------- CONFIG ----------------
TOP_K = 5  # Number of paragraphs to retrieve
EMBEDDINGS_FILE = "data/embeddings.pkl"
FAISS_INDEX_FILE = "data/faiss.index"

st.title("📝 ThesisBot - PDF QA & Summarization")

# PDF upload
pdf_file = st.file_uploader("Choose a PDF to analyze", type=["pdf"])

if pdf_file is not None:
    st.info("Extracting text from PDF...")
    paragraphs = extract_from_pdf(pdf_file)
    st.success(f"{len(paragraphs)} sentences extracted from the PDF.")

    # Embedding + FAISS
    if Path(EMBEDDINGS_FILE).exists() and Path(FAISS_INDEX_FILE).exists():
        st.info("Loading existing embeddings and FAISS index...")
        embeddings = load_embeddings(EMBEDDINGS_FILE)
        faiss_index = load_faiss_index(FAISS_INDEX_FILE)
    else:
        st.info("Creating embeddings and FAISS index...")
        embeddings = embed_paragraphs(paragraphs)
        faiss_index = build_faiss_index(embeddings)
        st.success("Embeddings and FAISS index created.")

    # Question input
    query = st.text_input("Enter your question:")

    if query:
        st.info("Retrieving relevant paragraphs...")
        top_paragraphs = retrieve_top_paragraphs(query, paragraphs, embeddings, faiss_index, top_k=TOP_K)

        st.subheader("Retrieved Relevant Paragraphs")
        for i, p in enumerate(top_paragraphs):
            st.write(f"{i+1}. {p[:200]}...")

        # Summarization
        st.info("Generating summary...")
        summary = synthesize_paragraphs(top_paragraphs)

        st.subheader("Generated Summary")
        st.write(summary)

        # Coverage evaluation
        coverage = coverage_score(summary, top_paragraphs)
        st.info(f"Coverage score: {coverage:.2f}")

        # Export PDF
        export_path = "output/summary.pdf"
        export_pdf(summary, export_path)
        st.success(f"Summary exported to `{export_path}`")
