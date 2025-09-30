from openai import OpenAI
client = OpenAI(api_key="TON_OPENAI_API_KEY")

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
