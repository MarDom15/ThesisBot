import streamlit as st
from pipeline.extractor import load_articles
from pipeline.vectorizer import embed_paragraphs, build_faiss_index
from agent.qa_agent import retrieve_top_paragraphs, reformulate_text
from agent.summarizer import synthesize_paragraphs
from evaluation.evaluate import coverage_score
from utils.file_utils import export_pdf

st.title("📝 Thesis Assistant AI Optimisé")

# Upload d'articles
uploaded_files = st.file_uploader("Téléversez vos articles (PDF/TXT)", accept_multiple_files=True)

articles = {}
if uploaded_files:
    import fitz
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

paragraphs = []
sources = []
for title, paras in articles.items():
    for p in paras:
        if len(p.strip()) > 20:
            paragraphs.append(p)
            sources.append(title)

# Embeddings
embeddings = embed_paragraphs(paragraphs)
index = build_faiss_index(embeddings)

# Inputs utilisateur
subject = st.text_input("Sujet de la thèse / Contexte :")
question = st.text_input("Pose ta question :")

if st.button("Obtenir réponse"):
    top_paras = retrieve_top_paragraphs(question, paragraphs, embeddings, index, subject=subject)
    
    st.subheader("Passages pertinents avec reformulation :")
    for i, p in enumerate(top_paras):
        st.write("**Original :**", p)
        st.write("**Reformulé :**", reformulate_text(p))
        st.write("---")
    
    st.subheader("Synthèse globale :")
    synth = synthesize_paragraphs(top_paras, sources[:len(top_paras)])
    st.write(synth)
    
    st.subheader("Score de couverture :")
    score = coverage_score(synth, top_paras)
    st.write(f"{score:.2f}")
    
    if st.button("Exporter synthèse PDF"):
        export_pdf(synth, "synthese_these.pdf")
        st.success("PDF exporté !")
