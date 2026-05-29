from flask import Flask, request, jsonify, render_template, g
from werkzeug.utils import secure_filename
from functools import wraps
import tempfile
import redis
import json
import math
import os
import re
import pymupdf
import numpy as np
import logging
import threading
import uuid
from dotenv import load_dotenv
from groq import Groq
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client, Client

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_url_path='', static_folder='.')

# LLM via Groq (free tier)
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')

COLLECTION = 'documents'

# Fail fast if required env vars are missing
_required = ['SUPABASE_URL', 'SUPABASE_ANON_KEY', 'SUPABASE_SERVICE_KEY', 'GROQ_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
_missing = [v for v in _required if not os.getenv(v)]
if _missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(_missing)}")

# Supabase clients
supabase_anon: Client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_ANON_KEY'))
supabase_admin: Client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY'))

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

# Semantic models — loaded lazily on first use to reduce startup memory
_embedding_model = None
_reranker_model = None
_model_lock = threading.Lock()


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                logging.info("Loading embedding model via fastembed...")
                _embedding_model = TextEmbedding(
                    model_name='sentence-transformers/all-MiniLM-L6-v2',
                    cache_dir=os.getenv('FASTEMBED_CACHE_PATH', None)
                )
    return _embedding_model


def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        with _model_lock:
            if _reranker_model is None:
                logging.info("Loading reranker via fastembed...")
                _reranker_model = TextCrossEncoder(
                    model_name='Xenova/ms-marco-MiniLM-L-6-v2',
                    cache_dir=os.getenv('FASTEMBED_CACHE_PATH', None)
                )
    return _reranker_model


# Qdrant Cloud vector store
qdrant = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
if not qdrant.collection_exists(COLLECTION):
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    for field in ('source', 'user_id'):
        qdrant.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD
        )
    logging.info(f"Created Qdrant collection: {COLLECTION}")
else:
    # Ensure user_id index exists on pre-existing collection
    try:
        qdrant.create_payload_index(
            collection_name=COLLECTION,
            field_name='user_id',
            field_schema=PayloadSchemaType.KEYWORD
        )
    except Exception:
        pass

# Sentence-boundary-aware chunker with overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# BM25 sparse index — per-user dicts, rebuilt from Qdrant on startup/upload
bm25_indices = {}   # user_id -> BM25Okapi
bm25_corpora = {}   # user_id -> list of (point_id, display_text)
bm25_lock = threading.Lock()
_bm25_ready = False


# ---- Auth ----

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized'}), 401
        token = auth_header[7:]
        try:
            user_response = supabase_admin.auth.get_user(token)
            if not user_response.user:
                return jsonify({'error': 'Unauthorized'}), 401
            g.user_id = str(user_response.user.id)
        except Exception:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated


# ---- Indexing ----

def rebuild_bm25_for_user(user_id):
    records, _ = qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key='user_id', match=MatchValue(value=user_id))]),
        with_payload=True,
        limit=10000
    )
    if not records:
        with bm25_lock:
            bm25_indices.pop(user_id, None)
            bm25_corpora.pop(user_id, None)
        return

    texts = [r.payload.get('text', '') for r in records]
    ids = [str(r.id) for r in records]
    tokenized = [t.lower().split() for t in texts]

    with bm25_lock:
        bm25_indices[user_id] = BM25Okapi(tokenized)
        bm25_corpora[user_id] = list(zip(ids, texts))
    logging.info(f"BM25 rebuilt for user {user_id[:8]}... with {len(texts)} chunks")


def rebuild_all_bm25():
    """Scroll all Qdrant records and rebuild per-user BM25 indices."""
    all_records = []
    offset = None
    while True:
        batch, offset = qdrant.scroll(
            collection_name=COLLECTION, with_payload=True, limit=1000, offset=offset
        )
        all_records.extend(batch)
        if offset is None:
            break

    user_chunks = {}
    for r in all_records:
        uid = r.payload.get('user_id')
        if uid:
            user_chunks.setdefault(uid, []).append(r)

    with bm25_lock:
        bm25_indices.clear()
        bm25_corpora.clear()

    for uid, records in user_chunks.items():
        texts = [r.payload.get('text', '') for r in records]
        ids = [str(r.id) for r in records]
        tokenized = [t.lower().split() for t in texts]
        with bm25_lock:
            bm25_indices[uid] = BM25Okapi(tokenized)
            bm25_corpora[uid] = list(zip(ids, texts))
    logging.info(f"BM25 rebuilt for {len(user_chunks)} user(s)")


def preprocess(text):
    text = text.replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def pdf_to_pages(path):
    try:
        doc = pymupdf.open(path)
        pages = []
        for page_num in range(doc.page_count):
            text = preprocess(doc[page_num].get_text('text'))
            if text:
                pages.append((page_num + 1, text))
        doc.close()
        return pages
    except Exception as e:
        logging.error(f"Error reading PDF {path}: {e}")
        return []


def index_pdf(pdf_path, user_id, force=False):
    """Extract, chunk, embed, and upsert a PDF into Qdrant for a specific user."""
    filename = os.path.basename(pdf_path)

    if force:
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=Filter(must=[
                FieldCondition(key='source', match=MatchValue(value=filename)),
                FieldCondition(key='user_id', match=MatchValue(value=user_id))
            ])
        )
        logging.info(f"Deleted existing chunks for {filename} (user {user_id[:8]}...)")
    else:
        existing, _ = qdrant.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key='source', match=MatchValue(value=filename)),
                FieldCondition(key='user_id', match=MatchValue(value=user_id))
            ]),
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
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}__{filename}__p{page_num}__c{chunk_idx}"))
            display_text = f"[Page {page_num}, Source: {filename}] {chunk}"
            embed_texts.append(chunk)
            display_texts.append(display_text)
            points.append((point_id, filename, page_num, display_text))

    if not points:
        return 0

    vecs = list(get_embedding_model().embed(embed_texts))

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload={'source': filename, 'page': page_num, 'text': display_text, 'user_id': user_id}
            )
            for (point_id, filename, page_num, display_text), vec in zip(points, vecs)
        ]
    )
    logging.info(f"Indexed {len(points)} chunks from {filename} for user {user_id[:8]}...")
    return len(points)


def init_index():
    global _bm25_ready
    rebuild_all_bm25()
    _bm25_ready = True
    logging.info("Startup init complete.")


# ---- Retrieval ----

def get_collection_count(user_id):
    try:
        result = qdrant.count(
            collection_name=COLLECTION,
            count_filter=Filter(must=[FieldCondition(key='user_id', match=MatchValue(value=user_id))])
        )
        return result.count
    except Exception:
        return 0


def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query, user_id, top_k=20):
    """Combine dense (Qdrant) and sparse (BM25) retrieval via RRF for a single user."""
    count = get_collection_count(user_id)
    if count == 0:
        return []

    # Dense retrieval via Qdrant, filtered to this user
    query_vec = list(get_embedding_model().embed([query]))[0].tolist()
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        query_filter=Filter(must=[FieldCondition(key='user_id', match=MatchValue(value=user_id))]),
        limit=min(top_k, count),
        with_payload=True
    )
    dense_ids = [str(h.id) for h in hits]
    dense_doc_map = {str(h.id): h.payload.get('text', '') for h in hits}

    # Sparse retrieval via per-user BM25
    with bm25_lock:
        local_bm25 = bm25_indices.get(user_id)
        local_corpus = list(bm25_corpora.get(user_id, []))

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


def find_relevant_chunks(query, user_id, top_n=3):
    """Return [(score, text), ...] reranked for a specific user."""
    if get_collection_count(user_id) == 0:
        return []

    candidates = hybrid_search(query, user_id, top_k=20)
    if not candidates:
        return []

    texts = [text for _, text in candidates]
    scores = list(get_reranker_model().rerank(query, texts))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(_sigmoid(float(score)), texts[idx]) for idx, score in ranked[:top_n]]


# ---- LLM ----

def generate_text(prompt):
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a helpful assistant that answers questions based on '
                        'provided document excerpts. Always cite page numbers when referencing '
                        'information. If the answer is not found in the provided context, '
                        'say so clearly rather than guessing.'
                    )
                },
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=750,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Groq API error: {e}", exc_info=True)
        return None


def ask_file(question, user_id):
    """Return (response_text, sources) for a specific user's documents."""
    scored_chunks = find_relevant_chunks(question, user_id, top_n=3)
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
    if response is None:
        return None, []
    return response, sources


# ---- Cache ----

def get_cached_response(cache_key):
    """Return (response, sources) from Redis, or (None, None) on miss/error."""
    if not redis_client:
        return None, None
    try:
        cached = redis_client.get(cache_key)
        if not cached:
            return None, None
        data = json.loads(cached)
        return data.get('response'), data.get('sources', [])
    except Exception:
        return None, None


def cache_response(cache_key, response, sources, ttl=3600):
    if not redis_client:
        return
    try:
        redis_client.setex(cache_key, ttl, json.dumps({'response': response, 'sources': sources}))
    except Exception:
        pass


# ---- Startup ----

threading.Thread(target=init_index, daemon=True).start()


# ---- Routes ----

@app.route('/')
def index():
    return render_template(
        'index.html',
        supabase_url=os.getenv('SUPABASE_URL', ''),
        supabase_anon_key=os.getenv('SUPABASE_ANON_KEY', '')
    )


@app.route('/upload', methods=['POST'])
@require_auth
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['pdf']
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Please upload a valid PDF file'}), 400

    filename = secure_filename(file.filename)
    user_id = g.user_id
    storage_path = f"{user_id}/{filename}"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    try:
        file.save(tmp.name)
        with open(tmp.name, 'rb') as f:
            file_bytes = f.read()

        # Upload to Supabase Storage (remove old version first if present)
        try:
            supabase_admin.storage.from_('pdfs').remove([storage_path])
        except Exception:
            pass
        supabase_admin.storage.from_('pdfs').upload(
            path=storage_path,
            file=file_bytes,
            file_options={'content-type': 'application/pdf'}
        )

        # Index to Qdrant
        count = index_pdf(tmp.name, user_id, force=True)
        rebuild_bm25_for_user(user_id)

        # Upsert documents record in Supabase
        supabase_admin.table('documents').delete().eq('user_id', user_id).eq('filename', filename).execute()
        supabase_admin.table('documents').insert({
            'user_id': user_id,
            'filename': filename,
            'storage_path': storage_path,
            'chunk_count': count
        }).execute()

        return jsonify({'message': f'Successfully indexed {count} chunks from {filename}'})
    except Exception as e:
        logging.error(f"Upload error: {e}", exc_info=True)
        return jsonify({'error': 'Upload failed, please try again'}), 500
    finally:
        os.unlink(tmp.name)


@app.route('/documents', methods=['GET'])
@require_auth
def list_documents():
    try:
        res = (
            supabase_admin.table('documents')
            .select('filename,chunk_count')
            .eq('user_id', g.user_id)
            .order('created_at')
            .execute()
        )
        documents = [{'filename': row['filename'], 'chunks': row['chunk_count']} for row in res.data]
        return jsonify({'documents': documents})
    except Exception as e:
        logging.error(f"Error in /documents: {e}")
        return jsonify({'documents': []}), 500


@app.route('/history', methods=['GET'])
@require_auth
def history():
    try:
        res = (
            supabase_admin.table('query_history')
            .select('id,question,answer,sources,created_at')
            .eq('user_id', g.user_id)
            .order('created_at', desc=True)
            .limit(50)
            .execute()
        )
        return jsonify({'history': res.data})
    except Exception as e:
        logging.error(f"Error in /history: {e}")
        return jsonify({'history': []}), 500


@app.route('/health')
def health():
    qdrant_ok = False
    try:
        qdrant.get_collections()
        qdrant_ok = True
    except Exception as e:
        logging.warning(f"Qdrant health check failed: {e}")
    status = {
        'status': 'ok' if (qdrant_ok and _bm25_ready) else 'starting',
        'qdrant_ok': qdrant_ok,
        'bm25_ready': _bm25_ready,
    }
    return jsonify(status), 200


@app.route('/ask', methods=['POST'])
@require_auth
def ask():
    try:
        question = request.form.get('question', '').strip()
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        user_id = g.user_id
        cache_key = f"{user_id}:{question}"

        response, sources = get_cached_response(cache_key)
        if response is None:
            response, sources = ask_file(question, user_id)
            if response is not None and 'No documents' not in response:
                cache_response(cache_key, response, sources)
                try:
                    supabase_admin.table('query_history').insert({
                        'user_id': user_id,
                        'question': question,
                        'answer': response,
                        'sources': sources
                    }).execute()
                except Exception as e:
                    logging.warning(f"Failed to save query history: {e}")

        if response is None:
            return jsonify({'error': 'Failed to generate a response, please try again'}), 500

        return jsonify({'response': response, 'sources': sources})

    except Exception as e:
        logging.error(f"Error in /ask: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred, please try again'})


if __name__ == '__main__':
    app.run(debug=True)
