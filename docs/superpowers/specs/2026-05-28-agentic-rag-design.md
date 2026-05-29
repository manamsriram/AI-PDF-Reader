# Agentic RAG + Cloud Redis Design

**Date:** 2026-05-28  
**Status:** Approved for implementation

---

## Context

Current RAG pipeline is linear and passive:  
`question → retrieve top-3 chunks → generate answer`

Two problems:
1. **Poor retrieval for complex questions** — multi-part questions get one retrieval pass; chunks for part B may drown out chunks for part A.
2. **Bad chunks silently degrade answers** — no grading; irrelevant chunks fed directly to LLM.

Additionally, Redis runs as a local process inside the Render Docker container via host/port config. This means no persistence across restarts and no managed failover. Upstash provides free managed Redis accessible via a single URL.

**Goals:**
- Add Query Decomposition + Corrective RAG (CRAG) as an agentic layer on top of existing retrieval
- Replace local Redis with Upstash (cloud-managed, free tier)
- Add in-memory L1 cache to reduce Upstash command usage
- Zero new paid services

---

## Architecture Overview

```
User question
    │
    ▼
[L1 cache check] ──hit──▶ return cached answer
    │ miss
    ▼
[L2 Upstash Redis check] ──hit──▶ populate L1, return cached answer
    │ miss
    ▼
[Query Decomposer]  (1 Groq call)
    │ → [q1, q2, q3]  (or just [original] if simple)
    │
    ▼ (for each sub-query, in sequence)
[Hybrid Search] (existing: dense + BM25 + RRF + rerank)
    │
    ▼
[CRAG Grader]  (1 Groq call per sub-query)
    │ relevant chunks ──────────────────────────────▶ keep
    │ all irrelevant
    ▼
[Query Reformulator]  (1 Groq call)
    │ → rewritten query
    ▼
[Hybrid Search again]  (no re-grade; use whatever comes back)
    │
    ▼
[Merge all sub-query chunks]  (deduplicate by text)
    │
    ▼
[generate_text()]  (1 Groq call — existing function, unchanged)
    │
    ▼
[Store in L1 + L2 cache]  (single-turn only; multi-turn skipped)
    │
    ▼
Return answer + sources
```

---

## Component 1: Agentic RAG Pipeline

### New functions in `app.py`

**`decompose_query(question: str) → list[str]`**
- Calls `groq_client.chat.completions.create()` with decomposition prompt
- Returns list of sub-queries (1–3 items)
- If LLM call fails or returns invalid JSON: return `[question]` (safe fallback)
- Temperature: 0.0 (deterministic)
- Max tokens: 200

**`grade_chunks(query: str, chunks: list[str]) → tuple[list[str], list[str]]`**
- Returns `(relevant_chunks, irrelevant_chunks)`
- Calls Groq with grading prompt; parses JSON response
- If LLM call fails: treat all chunks as relevant (safe fallback — no regression)
- Temperature: 0.0, max tokens: 150

**`reformulate_query(query: str) → str`**
- Returns rewritten query string
- If LLM call fails: return original query
- Temperature: 0.1, max tokens: 100

**`ask_file_agentic(question, user_id, conversation_history=None) → (str, list)`**
- Orchestrates decomp → CRAG loop → synthesis
- Falls back to `ask_file()` if any unrecoverable error
- Deduplicates chunks across sub-queries by exact text match before synthesis

### System prompts (module-level constants)

```python
_DECOMPOSE_PROMPT = """Split this question into independent sub-questions, each answerable from separate document excerpts. If the question is simple, return just the original.

Rules:
- Maximum 3 sub-questions
- Each must be self-contained
- Return ONLY a JSON array of strings

Question: {question}"""

_GRADE_PROMPT = """Determine if each chunk is relevant to the query.

Return ONLY JSON: {{"relevant": [0, 2], "irrelevant": [1]}}
(numbers are 0-based chunk indices)

Query: {query}
Chunks:
{chunks}"""

_REFORMULATE_PROMPT = """Rewrite this query to use different terminology that might appear in technical documents. The original failed to retrieve relevant content.

Original: {query}
Rewritten query (return ONLY the query text):"""
```

### LLM call budget

| Scenario | Calls |
|----------|-------|
| Simple question, good retrieval | 2 (grade + synthesize) |
| Complex 3-part question, good retrieval | 5 (decomp + 3×grade + synthesize) |
| Complex + all reformulations triggered | 8 (decomp + 3×grade + 3×reformulate + synthesize) |

Groq free tier: 30 req/min on llama-3.3-70b. All scenarios well within limits.

---

## Component 2: Two-Tier Cache

### L1 — In-memory TTLCache

```python
from cachetools import TTLCache
import threading

_memory_cache = TTLCache(maxsize=100, ttl=1800)  # 30 min TTL
_memory_cache_lock = threading.Lock()
```

- Max 100 entries (∼covers last ~100 unique questions per instance lifetime)
- TTL: 30 minutes
- Thread-safe via lock
- Lost on restart (acceptable — L2 persists)

### L2 — Upstash Redis

Replace host/port config with URL-based connection:

```python
REDIS_URL = os.getenv('REDIS_URL')  # e.g. rediss://default:token@host.upstash.io:6379
redis_client = redis.from_url(REDIS_URL) if REDIS_URL else None
```

- Upstash free tier: 10,000 commands/day, 256MB
- L1 absorbs hot queries → Upstash commands reserved for cold-start and cross-restart hits
- TTL: 3600s (1 hour) — same as current

### Updated cache functions

**`get_cached_response(cache_key)`**
1. Check L1 (`_memory_cache`)
2. On L1 miss, check L2 (Upstash)
3. On L2 hit, populate L1 and return
4. On both miss, return `(None, None)`

**`cache_response(cache_key, response, sources, ttl=3600)`**
1. Write to L1
2. Write to L2

### Cache key rules (unchanged)
- Single-turn: `f"{user_id}:{question}"`
- Multi-turn session: no cache (context varies per turn)
- Agentic queries: cache FINAL merged answer, not intermediate sub-query results

---

## Files to Modify

| File | Change |
|------|--------|
| `app.py` | Add 4 new functions, update `/ask` to call `ask_file_agentic()`, update cache tier logic, switch Redis to `from_url()` |
| `requirements.txt` | Add `cachetools>=5.3.0` |
| `.env.example` | Remove `REDIS_HOST`/`REDIS_PORT`, add `REDIS_URL` |

**Unchanged:** `hybrid_search()`, `find_relevant_chunks()`, `generate_text()`, `ask_file()` (kept as fallback), all auth/upload/history routes, frontend, Dockerfile.

---

## Upstash Setup (one-time, free)

1. Create account at upstash.com (free, no credit card)
2. Create Redis database → copy `REDIS_URL` (`rediss://...`)
3. Set `REDIS_URL` in Render environment variables
4. Remove `REDIS_HOST` and `REDIS_PORT` from Render env vars

---

## Verification

1. **Unit:** `decompose_query("What is X?")` returns `["What is X?"]` (no split)
2. **Unit:** `grade_chunks(q, irrelevant_chunks)` returns empty relevant list
3. **Integration:** Ask a multi-part question → check logs for `[decomp]` lines showing sub-queries
4. **Cache:** Ask same question twice → second response has `from_cache: true` in logs; Upstash dashboard shows command count
5. **Fallback:** With `REDIS_URL` unset, app starts fine and logs "Redis not available"
6. **CRAG:** Upload a PDF, ask an off-topic question → logs show reformulation attempt
