# 🔬 ResearchAI

> An AI-powered research assistant for exploring, analyzing, questioning, and comparing research documents using Retrieval-Augmented Generation (RAG).

ResearchAI allows users to upload research papers, retrieve relevant evidence, ask natural-language questions, analyze individual documents, and compare multiple research papers.

The system grounds responses in retrieved research content instead of relying only on the language model's general knowledge.

---

## ✨ What It Does

- 📄 Upload multiple PDF research papers
- 🔎 Retrieve semantically relevant research passages
- 💬 Ask natural-language questions about uploaded documents
- 🧠 Generate answers grounded in retrieved evidence
- 📊 Analyze individual research documents
- 🔬 Compare selected research papers
- 📚 Display document and page-level references
- 🚫 Detect questions outside the available research context
- 🗃️ Store document embeddings using ChromaDB
- 🖥️ Provide an interactive Streamlit interface

---

## 🧠 Architecture

```text
Research Papers
      ↓
PDF Processing
      ↓
Text Extraction
      ↓
Document Chunking
      ↓
Embeddings
      ↓
ChromaDB
      ↓
User Question
      ↓
Semantic Retrieval
      ↓
Relevant Research Context
      ↓
LLM
      ↓
Grounded Response
```

---

## 🔬 Research Analysis

ResearchAI can generate structured insights from uploaded research documents, including:

- Research objective
- Methodology
- Key findings
- Limitations
- Research takeaways

---

## 📚 Multi-Document Comparison

Selected research documents can be compared across:

- Research focus
- Methodology
- Key findings
- Similarities
- Differences
- Overall conclusions

The comparison is based on relevant content retrieved from the selected documents.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Interface | Streamlit |
| RAG | LangChain |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace Embeddings |
| PDF Processing | PyPDF |
| Text Processing | Sentence Transformers |
| LLM | Groq API |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Frostmark1618/ResearchAI.git
cd ResearchAI
```

### 2. Create a virtual environment

```powershell
py -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Configure the API key

Set the required environment variable:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Do not commit API keys or other secrets to the repository.

### 6. Run the application

```powershell
python -m streamlit run app.py
```

The application will start locally through Streamlit.

---

## 📂 Project Structure

```text
ResearchAI/
│
├── app.py
├── requirements.txt
├── README.md
└── medresearch_db/
```

---

## 🎥 Demo

A short demonstration of ResearchAI in action:

[▶️ Watch the ResearchAI Demo](./demo.mp4)

The demo shows the application workflow from uploading research papers to retrieving relevant evidence and generating research-grounded responses.

## 🎯 Use Cases

ResearchAI can support:

- Academic research exploration
- Research paper analysis
- Literature review assistance
- Technical document exploration
- Multi-document comparison
- Evidence-based question answering

---

## ⚠️ Limitations

ResearchAI is designed to ground responses in the documents provided by the user.

The quality of the generated response depends on factors such as:

- Document quality
- Retrieval quality
- Chunking strategy
- Embedding quality
- LLM behaviour

If the required information cannot be found in the available research material, the system should avoid presenting unrelated information as document-grounded evidence.

---

## 🔮 Future Improvements

- Additional document formats
- Improved retrieval and reranking
- Citation export
- Research history
- Downloadable analysis reports
- Advanced multi-document comparison
- Improved evaluation of retrieval quality

---

## 👨‍💻 Author

**Riddhiman Adak**

B.Tech CSE (AI & ML)

[GitHub](https://github.com/Frostmark1618) •
[Portfolio](https://riddhi-s-vision.vercel.app) •
[LinkedIn](https://www.linkedin.com/in/riddhiman-adak-5b6336307/)

---

⭐ If you find ResearchAI useful, consider starring the repository.
