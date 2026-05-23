from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import sqlite3
import redis
import json
import math
import os
import re
import ssl
import pymupdf
import numpy as np
import logging
import threading
import uuid
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
logging.basicConfig(level=logging.INFO)
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__, static_url_path='', static_folder='.')

# LLM via Groq (free tier)
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')

DATABASE = 'chatbot.db'
PDF_FOLDER = 'pdfFolder'
COLLECTION = 'documents'

# Redis (optional — caching disabled gracefully if unavailable)
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)
try:
    redis_client.ping()
    logging.info("Redis connected")
except Exception:
    redis_client = None
    logging.warning("Redis not available — caching disabled")

# Semantic models (CPU-friendly, ~90MB total, downloaded once)
logging.info("Loading embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
logging.info("Loading reranker (ms-marco-MiniLM-L-6-v2)...")
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Qdrant Cloud vector store
qdrant = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
if not qdrant.collection_exists(COLLECTION):
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    qdrant.create_payload_index(
        collection_name=COLLECTION,
        field_name="source",
        field_schema=PayloadSchemaType.KEYWORD
    )
    logging.info(f"Created Qdrant collection: {COLLECTION}")

# Sentence-boundary-aware chunker with overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# BM25 sparse index (in-memory, rebuilt from Qdrant on startup/upload)
bm25_index = None
bm25_corpus = []   # list of (point_id, display_text)
bm25_lock = threading.Lock()


# ---- Indexing ----

def rebuild_bm25():
    global bm25_index, bm25_corpus
    records, _ = qdrant.scroll(collection_name=COLLECTION, with_payload=True, limit=10000)
    if not records:
        with bm25_lock:
            bm25_index = None
            bm25_corpus = []
        return

    texts = [r.payload.get("text", "") for r in records]
    ids = [str(r.id) for r in records]
    tokenized = [t.lower().split() for t in texts]

    with bm25_lock:
        bm25_index = BM25Okapi(tokenized)
        bm25_corpus = list(zip(ids, texts))
    logging.info(f"BM25 rebuilt with {len(bm25_corpus)} chunks")


def preprocess(text):
    text = text.replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def pdf_to_pages(path):
    try:
        doc = pymupdf.open(path)
        pages = []
        for page_num in range(doc.page_count):
            text = preprocess(doc[page_num].get_text("text"))
            if text:
                pages.append((page_num + 1, text))
        doc.close()
        return pages
    except Exception as e:
        logging.error(f"Error reading PDF {path}: {e}")
        return []


def index_pdf(pdf_path, force=False):
    """Extract, chunk, embed, and upsert a PDF into Qdrant. Returns chunk count."""
    filename = os.path.basename(pdf_path)

    if not force:
        existing, _ = qdrant.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=filename))]),
            limit=1
        )
        if existing:
            logging.info(f"Already indexed: {filename}")
            return 0

    pages = pdf_to_pages(pdf_path)
    if not pages:
        return 0

    points = []
    embed_texts = []
    display_texts = []

    for page_num, page_text in pages:
        chunks = text_splitter.split_text(page_text)
        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            # Deterministic UUID so re-indexing the same file is idempotent
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}__p{page_num}__c{chunk_idx}"))
            display_text = f"[Page {page_num}, Source: {filename}] {chunk}"
            embed_texts.append(chunk)       # embed content only, not the prefix
            display_texts.append(display_text)
            points.append((point_id, filename, page_num, display_text))

    if not points:
        return 0

    vecs = embedding_model.encode(embed_texts, batch_size=32, show_progress_bar=False)

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload={"source": filename, "page": page_num, "text": display_text}
            )
            for (point_id, filename, page_num, display_text), vec in zip(points, vecs)
        ]
    )
    logging.info(f"Indexed {len(points)} chunks from {filename}")
    return len(points)


def init_index():
    os.makedirs(PDF_FOLDER, exist_ok=True)
    for fname in os.listdir(PDF_FOLDER):
        if fname.endswith('.pdf'):
            index_pdf(os.path.join(PDF_FOLDER, fname))
    rebuild_bm25()


# ---- Retrieval ----

def get_collection_count():
    try:
        return qdrant.get_collection(COLLECTION).points_count
    except Exception:
        return 0


def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query, top_k=20):
    """Combine dense (Qdrant) and sparse (BM25) retrieval via RRF."""
    if get_collection_count() == 0:
        return []

    # Dense retrieval via Qdrant
    query_vec = embedding_model.encode([query])[0].tolist()
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=min(top_k, get_collection_count()),
        with_payload=True
    )
    dense_ids = [str(h.id) for h in hits]
    dense_doc_map = {str(h.id): h.payload.get("text", "") for h in hits}

    # Sparse retrieval via BM25
    with bm25_lock:
        local_bm25 = bm25_index
        local_corpus = list(bm25_corpus)

    sparse_ids = []
    if local_bm25 and local_corpus:
        scores = local_bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        sparse_ids = [local_corpus[i][0] for i in top_indices if scores[i] > 0]

    # RRF fusion
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids])

    corpus_map = {cid: text for cid, text in local_corpus}
    corpus_map.update(dense_doc_map)

    candidates = []
    for doc_id, _ in fused[:top_k]:
        text = corpus_map.get(doc_id)
        if text:
            candidates.append((doc_id, text))
    return candidates


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def find_relevant_chunks(query, top_n=3):
    """Return [(score, text), ...] where score is a float in (0, 1), sigmoid-normalized from raw reranker logit."""
    if get_collection_count() == 0:
        return []

    candidates = hybrid_search(query, top_k=20)
    if not candidates:
        return []

    texts = [text for _, text in candidates]
    raw_scores = reranker_model.predict([(query, t) for t in texts])
    ranked = sorted(zip(raw_scores, texts), reverse=True)
    return [(_sigmoid(float(score)), text) for score, text in ranked[:top_n]]


# ---- LLM ----

def generate_text(prompt):
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions based on "
                        "provided document excerpts. Always cite page numbers when referencing "
                        "information. If the answer is not found in the provided context, "
                        "say so clearly rather than guessing."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=750,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Groq API error: {e}")
        return f"Error generating response: {e}"


def ask_file(question):
    """Return (response_text, sources) where sources is a list of dicts."""
    scored_chunks = find_relevant_chunks(question, top_n=3)
    if not scored_chunks:
        return "No documents have been indexed yet. Please upload a PDF first.", []

    sources = []
    prompt = (
        "Based on the following excerpts from documents, answer the question. "
        "Include page numbers when citing information. "
        "If the answer is not in the excerpts, say so.\n\n"
    )
    for score, text in scored_chunks:
        prompt += f"{text}\n\n"
        # Parse "[Page N, Source: filename] ..." prefix
        m = re.match(r'^\[Page (\d+), Source: ([^\]]+)\]\s*(.*)', text, re.DOTALL)
        if m:
            sources.append({
                'page': int(m.group(1)),
                'source': m.group(2),
                'text': m.group(3).strip(),
                'score': round(score, 4),
            })
        else:
            sources.append({'page': 0, 'source': 'unknown', 'text': text, 'score': round(score, 4)})

    prompt += f"Question: {question}\nAnswer:"
    response = generate_text(prompt)
    return response, sources


# ---- Database ----

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY, query TEXT, response TEXT)')
    conn.commit()
    conn.close()


def get_response_from_db(query):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT response FROM chat_history WHERE query = ?', (query,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def store_query_response(query, response):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('INSERT INTO chat_history (query, response) VALUES (?, ?)', (query, response))
    conn.commit()
    conn.close()


def get_cached_response(query):
    if not redis_client:
        return None
    try:
        cached = redis_client.get(query)
        return json.loads(cached) if cached else None
    except Exception:
        return None


def cache_response(query, response, ttl=3600):
    if not redis_client:
        return
    try:
        redis_client.setex(query, ttl, json.dumps(response))
    except Exception:
        pass


# ---- Startup ----

init_db()
init_index()


# ---- Routes ----

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['pdf']
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Please upload a valid PDF file'}), 400

    filename = secure_filename(file.filename)
    os.makedirs(PDF_FOLDER, exist_ok=True)
    save_path = os.path.join(PDF_FOLDER, filename)
    file.save(save_path)

    count = index_pdf(save_path, force=True)
    rebuild_bm25()

    return jsonify({'message': f'Successfully indexed {count} chunks from {filename}'})


@app.route('/documents', methods=['GET'])
def list_documents():
    try:
        all_records = []
        offset = None
        while True:
            batch, offset = qdrant.scroll(
                collection_name=COLLECTION,
                with_payload=True,
                limit=1000,
                offset=offset
            )
            all_records.extend(batch)
            if offset is None:
                break
        counts = {}
        for r in all_records:
            name = r.payload.get('source', 'unknown')
            counts[name] = counts.get(name, 0) + 1
        documents = [{'filename': name, 'chunks': count}
                     for name, count in sorted(counts.items())]
        return jsonify({'documents': documents})
    except Exception as e:
        logging.error(f"Error in /documents: {e}")
        return jsonify({'documents': []}), 500


@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.form.get('question', '').strip()
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        sources = []
        response = get_cached_response(question)
        if response is None:
            response = get_response_from_db(question)
            if not response:
                response, sources = ask_file(question)
                if response and "No documents" not in response:
                    store_query_response(question, response)
                    cache_response(question, response)

        return jsonify({'response': response, 'sources': sources})

    except Exception as e:
        logging.error(f"Error in /ask: {e}")
        return jsonify({'error': 'An error occurred, please try again'})


if __name__ == '__main__':
    app.run(debug=True)
