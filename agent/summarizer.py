import os
import pickle
from openai import OpenAI

# ------------------ CONFIG ------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("The environment variable OPENAI_API_KEY is not set.")
client = OpenAI(api_key=api_key)

# ------------------ LOAD PARAGRAPHS ------------------
try:
    with open("data/text_chunks.pkl", "rb") as f:
        paragraphs = pickle.load(f)  # list of text segments
except FileNotFoundError:
    raise FileNotFoundError("File 'data/text_chunks.pkl' not found.")

try:
    with open("data/article_map.pkl", "rb") as f:
        sources = pickle.load(f)     # list of sources corresponding to each segment
except FileNotFoundError:
    sources = None  # still works if no sources are available

# ------------------ SYNTHESIS FUNCTION ------------------
def synthesize_paragraphs(paragraphs, sources=None):
    combined = ""
    for i, p in enumerate(paragraphs):
        source = sources[i] if sources else "Unknown source"
        combined += f"[{source}]: {p}\n"

    prompt = (
        "Create a clear and concise summary of these passages, "
        "with academic rephrasing, keeping the source references:\n\n"
        f"{combined}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

# ------------------ INTERACTIVE LOOP ------------------
if __name__ == "__main__":
    total_segments = len(paragraphs)
    print(f"{total_segments} segments available for synthesis (from data\\text_chunks.pkl).")
    
    while True:
        user_input = input(
            "How many segments to synthesize? (enter 'all' for all, 0 to quit): "
        ).strip()

        if user_input == "0":
            print("Goodbye!")
            break
        elif user_input.lower() == "all":
            sample_paras = paragraphs
            sample_sources = sources
        else:
            try:
                n = int(user_input)
                if n > total_segments:
                    print(f"There are only {total_segments} segments. Synthesizing all.")
                    n = total_segments
                sample_paras = paragraphs[:n]
                sample_sources = sources[:n] if sources else None
            except ValueError:
                print("Invalid input. Enter a number, 'all', or 0 to quit.")
                continue

        print("\n--- Generating summary ---\n")
        try:
            summary = synthesize_paragraphs(sample_paras, sample_sources)
            print("\n--- Generated summary ---\n")
            print(summary)
        except Exception as e:
            print(f"[ERROR] Failed to generate summary: {e}")
2