import os
import pickle
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline

# ------------------ CONFIG ------------------
os.environ["HF_API_KEY"] = "YOUR_HF_API_KEY"   # Put your Hugging Face key if needed
MODEL_NAME = "tiiuae/falcon-rw-1b"

# Download NLTK tokenizers (for sentence segmentation)
nltk.download("punkt")
nltk.download("punkt_tab")

# HuggingFace pipeline
generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device=-1  # CPU; set device=0 if you have a GPU
)

# Data folder
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

text_chunks_path = os.path.join(data_dir, "text_chunks.pkl")
article_map_path = os.path.join(data_dir, "article_map.pkl")
summary_file = os.path.join(data_dir, "summary.txt")

# ------------------ PDF EXTRACTION ------------------
def extract_from_pdf(pdf_path):
    """Extracts text from a PDF and splits it into sentences."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # Split into sentences
    sentences = sent_tokenize(text)
    return sentences

# ------------------ CLEANING ------------------
def clean_paragraphs(paragraphs):
    """Cleans extracted segments to remove noise."""
    cleaned = []
    for p in paragraphs:
        p = p.strip()

        # Skip too short
        if len(p) < 25:
            continue

        # Skip DOI/URLs
        if p.lower().startswith("doi") or "http" in p:
            continue

        # Skip lines in all caps (often titles/noise)
        if p.isupper():
            continue

        # Skip if more digits than letters
        letters = sum(c.isalpha() for c in p)
        digits = sum(c.isdigit() for c in p)
        if digits > letters:
            continue

        cleaned.append(p)
    return cleaned

# ------------------ DATA SAVING ------------------
def prepare_data(pdf_path):
    """Creates text_chunks.pkl and article_map.pkl from a PDF."""
    print(f"[DEBUG] Extracting from {pdf_path}...")
    paragraphs = extract_from_pdf(pdf_path)
    paragraphs = clean_paragraphs(paragraphs)

    # Save segments
    with open(text_chunks_path, "wb") as f:
        pickle.dump(paragraphs, f)
    print(f"Total cleaned segments: {len(paragraphs)}")
    print("text_chunks.pkl created successfully!")

    # Map each segment to its source (PDF name)
    pdf_name = os.path.basename(pdf_path)
    sources = [pdf_name for _ in range(len(paragraphs))]
    with open(article_map_path, "wb") as f:
        pickle.dump(sources, f)
    print("article_map.pkl created successfully!")

    return paragraphs, sources

# ------------------ SYNTHESIS FUNCTION ------------------
def synthesize_paragraphs(paragraphs, sources=None):
    combined = ""
    for i, p in enumerate(paragraphs):
        source = sources[i] if sources else "Unknown source"
        combined += f"[{source}]: {p}\n"

    prompt = (
        "Create a clear and concise summary of these passages with academic rephrasing, "
        "keeping source references:\n\n" + combined
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

# ------------------ BATCH SUMMARIZATION ------------------
def batch_summarize(paragraphs, sources, batch_size=50):
    """Splits segments into batches to avoid overloading the model."""
    summaries = []
    for i in range(0, len(paragraphs), batch_size):
        chunk_paras = paragraphs[i:i+batch_size]
        chunk_sources = sources[i:i+batch_size]
        summary = synthesize_paragraphs(chunk_paras, chunk_sources)
        summaries.append(summary)
        print(f"[DEBUG] Batch {i//batch_size+1} summarized ({len(chunk_paras)} segments)")

    # Merge partial summaries
    final_summary = synthesize_paragraphs(
        summaries, ["Partial summary"] * len(summaries)
    )
    return final_summary

# ------------------ MAIN LOOP ------------------
if __name__ == "__main__":
    pdf_file = os.path.join(data_dir, "s42484-022-00083-z.pdf")

    if not os.path.exists(text_chunks_path) or not os.path.exists(article_map_path):
        paragraphs, sources = prepare_data(pdf_file)
    else:
        with open(text_chunks_path, "rb") as f:
            paragraphs = pickle.load(f)
        with open(article_map_path, "rb") as f:
            sources = pickle.load(f)
        print("✅ Existing files loaded from data/")

    print(f"{len(paragraphs)} segments ready for summarization.")

    # ⚡ Batch summarization
    summary = batch_summarize(paragraphs, sources, batch_size=50)

    print("\n--- Final Summary ---\n")
    print(summary)

    # Save to file
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\n✅ Final summary saved to {summary_file}")
