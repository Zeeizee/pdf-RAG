# PDF Q&A

Upload a PDF and ask questions about it. Answers come only from retrieved pages. If the file does not contain the answer, the app says so and skips sources.

**Stack:** Streamlit, LangChain, FAISS, OpenAI (`text-embedding-3-small` + `gpt-4o` / `gpt-4o-mini`)

## What it does

- Chunks the PDF and indexes it in FAISS (session memory)
- Shows a short overview after upload
- Streams grounded answers with **page + snippet** sources
- Sidebar: API key, model, chunk size, overlap, `top_k`

```
PDF → split → embed → FAISS → retrieve → stream answer (+ page sources)
```

## Run

Python 3.11+ and an [OpenAI API key](https://platform.openai.com/api-keys).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run main.py
```

Paste the key in the sidebar, or put `OPENAI_API_KEY=...` in a `.env` file (gitignored).

## Layout

```
main.py          UI
rag/ingest.py    load, chunk, FAISS
rag/chain.py     overview, retrieve, streaming
```

Text PDFs only (no OCR). One file per session. Refresh rebuilds the index.
