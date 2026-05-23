# DocSense AI

A Flask-based RAG app for chatting with PDF documents.

DocSense AI indexes PDF content into a vector database, combines dense + sparse retrieval, reranks relevant chunks, and generates grounded answers with citations.

## Current Stack

- **Backend:** Flask
- **LLM:** Groq API (`GROQ_MODEL`, default: `llama-3.3-70b-versatile`)
- **Vector DB:** Qdrant Cloud (`documents` collection)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Sparse retrieval:** BM25 (`rank-bm25`)
- **PDF parsing:** PyMuPDF
- **Caching / storage:** Redis (optional) + SQLite (`chatbot.db`)
- **Frontend:** Server-rendered template + static JS/CSS (Alpine.js)

## Features

- Upload and index PDFs via `/upload`
- Automatic page-aware chunking with overlap
- Hybrid retrieval (Qdrant dense search + BM25 sparse search via RRF)
- Cross-encoder reranking for final context selection
- Answers with source snippets, file names, pages, and confidence scores
- Redis + SQLite response reuse to reduce repeat latency
- Indexed document list endpoint (`/documents`)

## Repository Layout

- `/app.py` – Flask app, indexing, retrieval, LLM calls, and API routes
- `/templates/index.html` – main web UI
- `/static/js/app.js` – frontend behavior (upload, ask, sources, documents)
- `/static/css/app.css` – styling
- `/pdfFolder` – uploaded PDFs
- `/tests/test_app.py` – pytest tests
- `/vercel.json` – rewrite rules for frontend-to-backend API proxying
- `/Dockerfile` – containerized deployment

## API Endpoints

- `GET /` → web app
- `POST /upload` → upload a PDF (`pdf` form field)
- `GET /documents` → list indexed documents with chunk counts
- `POST /ask` → ask a question (`question` form field), returns `{ response, sources }`

## Local Setup

```bash
git clone https://github.com/manamsriram/DocSense-AI.git
cd DocSense-AI

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.3-70b-versatile
QDRANT_URL=https://your-qdrant-endpoint
QDRANT_API_KEY=your-qdrant-api-key

# Optional Redis cache
REDIS_HOST=localhost
REDIS_PORT=6379
```

Run the app:

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## Docker

Build and run:

```bash
docker build -t docsense-ai .
docker run --rm -p 10000:10000 --env-file .env docsense-ai
```

The container starts Gunicorn on port `10000`.

## Frontend + Backend Split Deployment

This repo includes `vercel.json` rewrites for `/ask`, `/upload`, and `/documents`.

Typical setup:

1. Deploy Flask backend (e.g., Render).
2. Deploy frontend files on Vercel.
3. Update `vercel.json` destinations to your backend URL.

## Tests

The repository contains pytest tests in `tests/test_app.py`.

If pytest is not installed in your environment:

```bash
pip install pytest
pytest -q
```

## Notes

- First startup can be slow because embedding/reranker models are loaded.
- Uploaded PDFs are indexed into the `documents` collection in Qdrant.
- Redis is optional; the app falls back gracefully when Redis is unavailable.
