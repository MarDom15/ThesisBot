# 📘 Quantum Research Assistant (QRA)  

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)

![Build](https://img.shields.io/badge/Status-Active-success.svg)

---

## 🌌 Introduction  

The **Quantum Research Assistant (QRA)** is an intelligent research tool designed to support **students, researchers, and professionals** working in advanced scientific domains.  

In today’s world, the **amount of scientific literature is growing exponentially**. Reading, analyzing, and synthesizing hundreds of papers is not only time-consuming but also cognitively demanding.  

QRA aims to **bridge the gap** by combining:  
- **Natural Language Processing** for reading and extracting knowledge from PDFs,  
- **Large Language Models** for generating academic reformulations and summaries,  
- **Vector Databases** for fast retrieval of relevant information,  
- **Interactive Dashboards** to make research exploration more intuitive.  

👉 Think of QRA as your **AI-powered co-pilot** for thesis writing, literature reviews, and scientific discovery.  

---

## 🚀 Features  

### 🔍 Reading & Extraction  
- Ingest **PDFs** (articles, books, theses).  
- Automatically extract and segment content into relevant sections.  

### ❓ Question Answering  
- Ask any research question → QRA finds answers in your sources.  
- Responses include:  
  - 📖 *Original citation* (exact source)  
  - ✍️ *Academic reformulation* (plagiarism-free)  
  - 📑 *Contextual summary*  

### 📑 Multi-source Synthesis  
- Generate **unique paragraphs** that merge insights from multiple sources.  
- Use of **LaTeX** for precise mathematical notation.  

### 🧠 Knowledge Organization  
- Build a **FAISS vector store** for lightning-fast retrieval.  
- Automatic clustering of documents by domain (`Quantum ML`, `Quantitative Finance`, `Stochastics`, etc.).  

### 📊 Research Dashboard  
- Streamlit-powered interface for exploration.  
- Search, filter, and interact with knowledge visually.  

---

## 🎯 Use Cases  

- 📚 **PhD Students** → Accelerate literature reviews, build state-of-the-art sections.  
- 👩‍🔬 **Researchers** → Quickly compare sources, extract definitions & formulas.  
- 🧑‍🏫 **Professors** → Prepare lecture notes & references efficiently.  
- 💼 **Finance & Data Scientists** → Apply quantum methods to pricing models and simulations.  

---

## 🛠️ Tech Stack  

- **Language**: Python 3.10+  
- **AI Framework**: [LangChain](https://www.langchain.com/)  
- **LLMs**: OpenAI GPT / Anthropic Claude / Mistral (configurable via `.env`)  
- **Vectorization**: FAISS + SentenceTransformers  
- **PDF Parsing**: PyMuPDF, pypdf  
- **Interface**: Streamlit (interactive dashboard)  
- **Math Support**: LaTeX rendering  

---

## 📂 Project Structure  

```
quantum-research-assistant/
│── data/               # PDFs (articles, books, theses)
│── notebooks/          # Jupyter explorations
│── src/
│   ├── ingestion.py    # Source reading & preprocessing
│   ├── embeddings.py   # FAISS vectorization
│   ├── agent.py        # Q&A specialized agent
│   ├── synthesis.py    # Multi-source synthesis
│   └── ui.py           # Streamlit interface
│── requirements.txt    # Project dependencies
│── README.md           # Project documentation
```

---

## ⚙️ Installation  

### 1. Clone the repository  
```bash
git clone https://github.com/MarDom15/ThesisBot
cd ThesisBot
```

### 2. Create a virtual environment  
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies  
```bash
pip install -r requirements.txt
```

### 4. Configure API keys  
Create a `.env` file at the project root:  

```ini
OPENAI_API_KEY=sk-xxxx
```

---

## 🖥️ Usage  

### 1. Add your sources  
Put your PDF files (articles, books, theses) into the `data/` folder.  

### 2. Build the vector database  
```bash
python src/ingestion.py
```

### 3. Launch the user interface  
```bash
streamlit run src/ui.py
```

### 4. Example Workflow  

**Question:**  
*"How does quantum machine learning improve option pricing?"*  

**Agent Response:**  

📖 **Citation**:  
*"Quantum algorithms can reduce the computational complexity of option pricing models compared to classical Monte Carlo simulations."*  
(Source: *Quantum Finance Journal, 2023*)  

✍️ **Reformulation**:  
Quantum algorithms reduce the computational complexity of option pricing models, providing a more efficient alternative to classical approaches such as Monte Carlo simulations.  

📑 **Synthesis**:  
In quantitative finance, integrating quantum ML can significantly accelerate option pricing, especially for complex portfolios.  

---

## 📊 Possible Applications  

- 📚 Thesis/dissertation writing support  
- 🧮 Extraction of formulas & key definitions  
- 🧠 Automatic synthesis across sources  
- 📑 Literature review preparation  
- 📝 Anti-plagiarism paraphrasing  

---

## 🔮 Roadmap  

- ✅ PDF ingestion & basic Q&A  
- ✅ Multi-source synthesis with LaTeX support  
- 🚧 Zotero/Mendeley integration for reference management  
- 🚧 Automatic **PowerPoint generation** from summaries  
- 🚧 FR ↔ EN scientific translation  
- 🚧 Advanced source comparison (similarities/differences)  

---

## 🤝 Contributing  

Contributions are welcome! 🚀  

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/new-feature`)  
3. Commit your changes (`git commit -m "Add new feature"`)  
4. Push to the branch (`git push origin feature/new-feature`)  
5. Open a Pull Request  

---

## 📖 License  

This project is licensed under the **Martial Domche** – free for academic and research use. 

---

## 📢 Contact  

- 🐦 Twitter: [@martialdomche](https://twitter.com/martialdomche)  
- 💼 LinkedIn: [Martial Domche](https://www.linkedin.com/martialdomche)  
- 📧 Email: [mdomche@gmail.com](mailto:mdomche@gmail.com)  

👉 QRA is designed to save researchers **time and effort**, while ensuring **academic rigor and source fidelity**.  
