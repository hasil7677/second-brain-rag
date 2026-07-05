"""
OKF Pipeline — FastAPI Server
Run: uvicorn app:app --reload --port 8001
"""
import json, asyncio, time
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import pipeline as pl

app = FastAPI(title="OKF Pipeline")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HERE = Path(__file__).parent


# ── pages ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return (HERE / "OKF_EXPLAINER.html").read_text(encoding="utf-8")

@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline_page():
    return (HERE / "index.html").read_text(encoding="utf-8")

@app.get("/graph", response_class=HTMLResponse)
async def graph_page():
    return (HERE / "graph.html").read_text(encoding="utf-8")


# ── corpus state ──────────────────────────────────────────────────────────────
@app.get("/api/corpus")
async def get_corpus():
    nodes, edges = [], []
    for slug, v in pl.corpus.items():
        nodes.append({
            "id": slug, "label": v["label"], "type": v["type"],
            "source": v["source"], "tags": v["tags"], "body": v["body"],
        })
        for link in v["links"]:
            if link in pl.corpus:
                edges.append({
                    "source": slug, "target": link,
                    "type": "cross_doc" if pl.corpus[link]["source"] != v["source"] else "wikilink",
                })
    return {"nodes": nodes, "edges": edges, "total": len(pl.corpus)}


# ── ingest SSE ────────────────────────────────────────────────────────────────
@app.get("/api/ingest")
async def ingest_stream():
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_event(evt_type, payload):
        loop.call_soon_threadsafe(queue.put_nowait, {"event": evt_type, "data": payload})

    async def run_ingest():
        await loop.run_in_executor(None, lambda: pl.ingest(on_event=on_event))
        await queue.put(None)  # sentinel

    asyncio.create_task(run_ingest())

    async def generator():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield {"event": item["event"], "data": json.dumps(item["data"])}

    return EventSourceResponse(generator())


# ── query ─────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

@app.post("/api/query")
async def query_endpoint(req: QueryRequest):
    if not pl.corpus:
        return JSONResponse({"error": "Corpus empty — run ingestion first"}, status_code=400)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: pl.query(req.question, verbose=False))
    return result


# ── health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "corpus_size": len(pl.corpus)}
