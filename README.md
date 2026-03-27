# Medical RAG Chatbot

A small full-stack demo that answers questions from your own medical PDFs. It uses **retrieval-augmented generation (RAG)** so the model grounds answers in documents you control—not generic training data alone.

**Live flow:** browser UI → Flask API → vector search (Pinecone) → OpenAI chat model → cited, context-aware replies.

---

## Highlights

- **RAG pipeline** with LangChain: chunk PDFs, embed locally, store in a vector index, retrieve top matches, then generate answers.
- **Flask** web app with a simple chat front end.
- **Pinecone** as the managed vector database for similarity search.
- **OpenAI** (`gpt-4o`) for generation; **Hugging Face / sentence-transformers** (`all-MiniLM-L6-v2`) for embeddings.
- Clear separation between **ingestion** (`store_index.py`) and **serving** (`app.py`).

---

## Tech stack

| Area | Technologies |
|------|----------------|
| Language | Python 3 |
| Web framework | Flask, Jinja2 templates |
| LLM orchestration | LangChain, LangChain OpenAI, LangChain Pinecone, LangChain Community |
| Embeddings | Hugging Face `sentence-transformers`, LangChain Hugging Face integrations |
| Vector database | Pinecone |
| Large language model | OpenAI API (GPT-4o) |
| Documents | PyPDF, PDF loading and text splitting |
| Configuration | python-dotenv |

*Keywords useful for resumes and ATS: Python, Flask, REST API, RAG, LangChain, LLM, OpenAI API, vector database, Pinecone, embeddings, semantic search, Hugging Face, PDF ingestion, NLP.*

---

## What you need

- Python **3.10+** (3.11 or 3.12 works well)
- A **Pinecone** account and an index (dimension **384** for `all-MiniLM-L6-v2`, metric e.g. cosine)
- An **OpenAI API key** with access to the model you configure in `app.py`

---

## Quick start

### 1. Clone and enter the project

```bash
git clone https://github.com/lavanesh-tech/Medical-RAG-System.git
cd Medical-RAG-System
```

### 2. Virtual environment (recommended)

```bash
python -m venv .venv
# Windows (Git Bash / PowerShell)
source .venv/Scripts/activate
# macOS / Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

*If you use Conda instead: create an environment with Python 3.10+, activate it, then run `pip install -r requirements.txt`.*

### 3. Environment variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key
```

Use the index name expected by the code (**`medical-chatbot`**) or change `index_name` in `app.py` and `store_index.py` to match your Pinecone index.

### 4. Load your PDFs into Pinecone

Put PDFs in a folder named **`data/`** next to `store_index.py`, then run:

```bash
python store_index.py
```

This reads PDFs, chunks text, embeds, and upserts vectors into your Pinecone index.

### 5. Run the app

```bash
python app.py
```

Open **http://127.0.0.1:8080** in your browser and chat.

---

## Project layout

| Path | Role |
|------|------|
| `app.py` | Flask server, RAG chain, chat route |
| `store_index.py` | One-shot ingestion: PDFs → chunks → Pinecone |
| `src/helper.py` | Embeddings and PDF utilities |
| `src/prompt.py` | System prompt for the assistant |
| `templates/chat.html` | Chat UI |
| `static/style.css` | Styling |
| `requirements.txt` | Python dependencies |

---

## Troubleshooting

- **Missing Python packages:** always activate the same venv you used for `pip install`, then run `pip install -r requirements.txt` again.
- **Pinecone / plugin errors:** use the `pinecone` package from this repo’s `requirements.txt`; avoid mixing deprecated `pinecone-client` + old plugins.
- **Empty or generic answers:** confirm ingestion finished successfully and the index name and embedding dimension match Pinecone.

---

## Author

**LAVANESH THIRUKONDA MAHENDRAN** — see `setup.py` / repository for contact details.

## License

See `LICENSE` in this repository.
