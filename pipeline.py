"""
OKF Pipeline — Standalone
=========================
Second brain for AI agents. Ingest markdown docs → extract concept files →
hybrid retrieval (BM25 + BGE-M3 + RRF) → wikilink enrichment → LLM answer.

Usage:
    python pipeline.py ingest                        # build corpus from okf_data/
    python pipeline.py query "your question here"   # query the corpus
"""

import os, json, re, sys, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import boto3

# ── config ────────────────────────────────────────────────────────────────────
REGION          = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
MISTRAL_SMALL   = "mistral.mistral-7b-instruct-v0:2"
MISTRAL_LARGE   = "mistral.mistral-large-2402-v1:0"
DOCS_DIR        = Path(__file__).parent.parent / "okf_data"
ENRICH_DEPTH    = 2   # how many wikilink hops to follow per top result

bedrock = boto3.client("bedrock-runtime", region_name=REGION)

# ── in-memory corpus ──────────────────────────────────────────────────────────
corpus: dict[str, dict] = {}   # slug → {title, body, tags, links, source, type}


# ── bedrock helpers ───────────────────────────────────────────────────────────
def _mistral(prompt: str, model: str = MISTRAL_SMALL, max_tokens: int = 600) -> str:
    body = json.dumps({
        "prompt": f"<s>[INST] {prompt} [/INST]",
        "max_tokens": max_tokens,
        "temperature": 0.15,
    })
    resp = bedrock.invoke_model(modelId=model, body=body)
    return json.loads(resp["body"].read()).get("outputs", [{}])[0].get("text", "").strip()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — INGEST
# ══════════════════════════════════════════════════════════════════════════════

EXTRACT_PROMPT = """You are an expert at extracting structured knowledge from insurance documents.

Read the document below and extract 6-8 key concepts as OKF (Open Knowledge Format) concept files.

Each concept must follow this EXACT format:
===CONCEPT===
SLUG: kebab-case-slug
TITLE: Human readable title
TYPE: one of [underwriting_rule, claims_procedure, risk_framework, product_coverage]
SUMMARY: 2-3 sentence summary of the concept
TAGS: comma-separated tags
RELATED: [[slug-1]], [[slug-2]]  (leave empty if none)
===END===

Rules:
- Extract only facts explicitly stated in the document
- SLUG must be kebab-case, unique, descriptive
- RELATED should reference other slugs you are extracting from THIS document
- Be specific — "flood-zone-2-eligibility" not "flood"

Document:
{content}

Extract concepts now:"""


def _parse_concepts(text: str, source_doc: str) -> list[dict]:
    concepts = []
    for block in re.findall(r"===CONCEPT===(.*?)===END===", text, re.DOTALL):
        c = {}
        for field in ["SLUG", "TITLE", "TYPE", "SUMMARY", "TAGS", "RELATED"]:
            m = re.search(rf"{field}:\s*(.+?)(?=\n[A-Z]+:|$)", block, re.DOTALL)
            c[field.lower()] = m.group(1).strip() if m else ""
        if c.get("slug") and c.get("title"):
            c["source"] = source_doc
            c["tags"]   = [t.strip() for t in c.get("tags", "").split(",") if t.strip()]
            c["links"]  = re.findall(r"\[\[([^\]]+)\]\]", c.get("related", ""))
            concepts.append(c)
    return concepts


def ingest(docs_dir: Path = DOCS_DIR, on_event=None) -> None:
    """
    on_event(event_type, payload) — optional callback for SSE streaming.
    event_type: "doc_start" | "concept" | "edge" | "doc_done" | "done" | "error"
    """
    global corpus
    corpus = {}

    def _emit(evt, payload):
        if on_event:
            on_event(evt, payload)

    md_files = list(docs_dir.glob("*.md"))
    if not md_files:
        _emit("error", {"message": f"No .md files found in {docs_dir}"})
        return

    _emit("start", {"total_docs": len(md_files)})
    print(f"\n\U0001f4da Ingesting {len(md_files)} documents from {docs_dir}\n")

    for md_path in md_files:
        source = md_path.stem
        content = md_path.read_text(encoding="utf-8")
        _emit("doc_start", {"source": source, "filename": md_path.name, "chars": len(content)})
        print(f"  ⚙️  Extracting concepts from {md_path.name} ({len(content)} chars)...")

        chunks = [content[i:i+6000] for i in range(0, min(len(content), 18000), 6000)]
        all_concepts = []
        for chunk in chunks:
            raw = _mistral(EXTRACT_PROMPT.format(content=chunk), model=MISTRAL_LARGE, max_tokens=1200)
            all_concepts.extend(_parse_concepts(raw, source))

        seen = set()
        for c in all_concepts:
            slug = f"{source}--{c['slug']}"
            if slug not in seen:
                seen.add(slug)
                entry = {
                    "title":  c["slug"],
                    "label":  c["title"],
                    "type":   c.get("type", "concept"),
                    "body":   c["summary"],
                    "tags":   c["tags"],
                    "links":  [f"{source}--{l}" for l in c["links"]],
                    "source": source,
                }
                corpus[slug] = entry
                _emit("concept", {
                    "slug": slug, "label": entry["label"], "type": entry["type"],
                    "source": source, "tags": entry["tags"], "body": entry["body"],
                })
                print(f"      ✓ {c['title']}")

        _emit("doc_done", {"source": source, "count": sum(1 for v in corpus.values() if v["source"] == source)})

    # emit edges after all concepts exist
    for slug, v in corpus.items():
        for link in v["links"]:
            if link in corpus:
                edge_type = "cross_doc" if corpus[link]["source"] != v["source"] else "wikilink"
                _emit("edge", {"source": slug, "target": link, "type": edge_type})

    _emit("done", {"total": len(corpus)})
    print(f"\n✅ Corpus built: {len(corpus)} concept files\n")
    _print_graph_summary()


def _print_graph_summary() -> None:
    total_edges = sum(len(v["links"]) for v in corpus.values())
    # cross-doc edges
    cross = sum(
        1 for v in corpus.values() for link in v["links"]
        if link in corpus and corpus[link]["source"] != v["source"]
    )
    print(f"  📊 Graph: {len(corpus)} nodes · {total_edges} edges · {cross} cross-document edges")
    # tag index
    tag_index: dict[str, list] = {}
    for slug, v in corpus.items():
        for tag in v["tags"]:
            tag_index.setdefault(tag, []).append(slug)
    cross_tag = sum(
        1 for slugs in tag_index.values()
        if len({corpus[s]["source"] for s in slugs}) > 1
    )
    print(f"  🏷️  Tags: {len(tag_index)} unique · {cross_tag} shared across documents\n")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — ROUTE  (cheap Mistral-7B triage)
# ══════════════════════════════════════════════════════════════════════════════

def route(question: str) -> dict:
    titles = ", ".join(v["label"] for v in corpus.values())
    prompt = f"""You are a query router for an OKF knowledge base.

Corpus concepts: {titles}

Query: "{question}"

IN_CORPUS  — answer is a specific named fact/rule in the curated concept files
HYBRID_RETRIEVAL — answer needs broad search across many files

Respond ONLY with JSON:
{{"path": "IN_CORPUS" or "HYBRID_RETRIEVAL", "reasoning": "one sentence", "confidence": 0.0-1.0, "key_entities": ["entity1"]}}"""

    try:
        raw = _mistral(prompt, model=MISTRAL_SMALL, max_tokens=150)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group()) if m else {"path": "HYBRID_RETRIEVAL", "reasoning": "parse error", "confidence": 0.5, "key_entities": []}
    except Exception as e:
        return {"path": "HYBRID_RETRIEVAL", "reasoning": str(e), "confidence": 0.5, "key_entities": []}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — RETRIEVE  (BM25 + BGE-M3 + RRF)
# ══════════════════════════════════════════════════════════════════════════════

_bge_model = None

def _get_bge():
    global _bge_model
    if _bge_model is None:
        print("  🔄 Loading BGE-M3 embedding model (first query only)...")
        from sentence_transformers import SentenceTransformer
        _bge_model = SentenceTransformer("BAAI/bge-m3")
        print("  ✅ BGE-M3 ready\n")
    return _bge_model


def hybrid_retrieve(question: str, top_k: int = 5) -> list[dict]:
    from rank_bm25 import BM25Okapi
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    items  = list(corpus.items())
    docs   = [f"{v['label']} {v['body']} {' '.join(v['tags'])}" for _, v in items]

    # BM25
    bm25        = BM25Okapi([d.lower().split() for d in docs])
    bm25_scores = bm25.get_scores(question.lower().split())

    # BGE-M3 semantic
    try:
        model      = _get_bge()
        q_emb      = model.encode([question], normalize_embeddings=True)
        doc_embs   = model.encode(docs, normalize_embeddings=True)
        sem_scores = cosine_similarity(q_emb, doc_embs)[0]
    except Exception:
        sem_scores = np.zeros(len(docs))

    # RRF  k=60
    k         = 60
    bm25_rank = {i: r for r, i in enumerate(np.argsort(-bm25_scores))}
    sem_rank  = {i: r for r, i in enumerate(np.argsort(-sem_scores))}
    rrf       = {i: 1/(k + bm25_rank[i]) + 1/(k + sem_rank[i]) for i in range(len(items))}

    top = sorted(rrf, key=lambda x: -rrf[x])[:top_k]
    return [
        {
            "slug":       items[i][0],
            "title":      items[i][1]["label"],
            "type":       items[i][1]["type"],
            "body":       items[i][1]["body"],
            "tags":       items[i][1]["tags"],
            "links":      items[i][1]["links"],
            "source":     items[i][1]["source"],
            "bm25_score": round(float(bm25_scores[i]), 3),
            "sem_score":  round(float(sem_scores[i]),  3),
            "rrf_score":  round(rrf[i], 4),
            "via":        "hybrid",
        }
        for i in top
    ]


def corpus_retrieve(question: str, key_entities: list[str], top_k: int = 5) -> list[dict]:
    scored = []
    q_words = set(question.lower().split())
    for slug, v in corpus.items():
        score = 0
        tl, bl = v["label"].lower(), v["body"].lower()
        for ent in key_entities:
            if ent.lower() in tl: score += 3
            if ent.lower() in bl: score += 1
        if q_words & set(tl.split()): score += 1
        if score > 0:
            scored.append((score, slug, v))
    scored.sort(reverse=True)
    return [
        {"slug": s, "title": v["label"], "type": v["type"], "body": v["body"],
         "tags": v["tags"], "links": v["links"], "source": v["source"],
         "score": sc, "via": "corpus"}
        for sc, s, v in scored[:top_k]
    ]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — ENRICH  (follow wikilinks from top results)
# ══════════════════════════════════════════════════════════════════════════════

def enrich(results: list[dict], depth: int = ENRICH_DEPTH) -> list[dict]:
    """
    Follow [[wikilinks]] from each retrieved concept up to `depth` hops.
    Adds linked concept bodies to the context — the LLM sees not just the
    retrieved concept but its connected knowledge too.
    """
    seen_slugs = {r["slug"] for r in results}
    enriched   = list(results)

    for result in results[:3]:   # enrich from top-3 results only
        for link_slug in result["links"][:depth]:
            if link_slug in corpus and link_slug not in seen_slugs:
                linked = corpus[link_slug]
                enriched.append({
                    "slug":   link_slug,
                    "title":  linked["label"],
                    "type":   linked["type"],
                    "body":   linked["body"],
                    "tags":   linked["tags"],
                    "links":  linked["links"],
                    "source": linked["source"],
                    "via":    "wikilink_enrichment",
                    "enriched_from": result["slug"],
                })
                seen_slugs.add(link_slug)

    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — GENERATE
# ══════════════════════════════════════════════════════════════════════════════

def generate(question: str, sources: list[dict]) -> str:
    context_parts = []
    for s in sources:
        via_note = f" [via {s['via']}]" if s.get("via") else ""
        context_parts.append(f"## {s['title']} ({s['type']}){via_note}\n{s['body']}")
    context = "\n\n".join(context_parts)

    prompt = f"""You are an insurance AI assistant. Answer ONLY from the context below.
If the context does not contain enough information, say INSUFFICIENT_CONTEXT.

Context:
{context}

Question: {question}

Answer concisely, citing specific rules or numbers from the context:"""

    return _mistral(prompt, model=MISTRAL_LARGE, max_tokens=400)


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE QUERY
# ══════════════════════════════════════════════════════════════════════════════

def query(question: str, verbose: bool = True) -> dict:
    if not corpus:
        return {"error": "Corpus is empty — run ingest first", "answer": None}

    t0 = time.time()

    # 1. Route
    route_result = route(question)
    path         = route_result["path"]
    if verbose:
        print(f"\n🔀 Router → {path} (conf {route_result['confidence']:.0%})")
        print(f"   Reasoning: {route_result['reasoning']}")

    # 2. Retrieve
    if path == "IN_CORPUS":
        results = corpus_retrieve(question, route_result.get("key_entities", []))
    else:
        results = hybrid_retrieve(question)

    if verbose:
        print(f"\n📄 Retrieved {len(results)} concepts:")
        for r in results:
            score_str = f"rrf={r['rrf_score']}" if "rrf_score" in r else f"score={r.get('score', '?')}"
            print(f"   • {r['title']} [{r['source']}] {score_str}")

    # 3. Enrich via wikilinks
    enriched = enrich(results)
    new_nodes = [e for e in enriched if e["via"] == "wikilink_enrichment"]
    if verbose and new_nodes:
        print(f"\n🔗 Wikilink enrichment added {len(new_nodes)} connected concepts:")
        for n in new_nodes:
            print(f"   • {n['title']} ← from {n['enriched_from']}")

    # 4. Generate
    answer = generate(question, enriched)

    # 5. Fallback if insufficient
    fallback = False
    if answer.startswith("INSUFFICIENT_CONTEXT") and path == "IN_CORPUS":
        if verbose:
            print("\n⚠️  IN_CORPUS insufficient — falling back to hybrid...")
        results  = hybrid_retrieve(question)
        enriched = enrich(results)
        answer   = generate(question, enriched)
        fallback = True

    latency = round(time.time() - t0, 2)

    if verbose:
        print(f"\n💬 Answer ({latency}s):\n{answer}\n")
        print(f"   Path: {path}{'→HYBRID_FALLBACK' if fallback else ''}")
        print(f"   Sources used: {len(enriched)} ({len(new_nodes)} via wikilinks)\n")

    return {
        "answer":            answer,
        "path":              f"{path}→HYBRID_FALLBACK" if fallback else path,
        "sources":           enriched,
        "enriched_count":    len(new_nodes),
        "latency_s":         latency,
        "fallback":          fallback,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "ingest":
        ingest()

    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Usage: python pipeline.py query \"your question\"")
            sys.exit(1)
        # auto-ingest if corpus is empty
        ingest()
        query(sys.argv[2])

    elif cmd == "demo":
        ingest()
        questions = [
            "Can we write flood cover for a Zone 2 property?",
            "What triggers an automatic declination under underwriting guidelines?",
            "When does a claims consultant need to appoint a loss adjustor?",
            "What perils are excluded from all commercial property policies?",
        ]
        for q in questions:
            print("─" * 60)
            query(q)

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
