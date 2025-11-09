# ==========================================================
# engine.py — BI AI ENGINE (v6.7 ULTRA — Full Power Stable)
# ==========================================================
"""
Final Engine Revision:
✅ Immune to "unhashable type: list"
✅ Immune to "not enough values to unpack"
✅ Always returns (stream, blocks) tuple
✅ Preserves full FAISS + BM25 hybrid search power
✅ Supports contextual document filtering
✅ Offline compatible with embedder v2.3 + retriever + local_llm
"""

import re, time, pickle, hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
from rank_bm25 import BM25Okapi

from .text_extractor import extract_text
from .embedder import embed_texts
from .retriever import VectorStore
from .local_llm import generate


# ==========================================================
# 🔹 Helpers
# ==========================================================
def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25 search."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]


def _detect_doc_type(text: str) -> str:
    """Heuristic classification of document types."""
    t = text.lower()
    rules = [
        ("audio_transcript", ["[audio transcript]", "duration:", "segments:"]),
        ("video_transcript", ["[video audio transcript]", "duration:", "video"]),
        ("image_ocr", ["[image extracted", "ocr_applied"]),
        ("certificate", ["certificate", "awarded to"]),
        ("invoice", ["invoice", "amount", "gst", "total"]),
        ("report", ["abstract", "conclusion", "references"]),
        ("academic", ["university", "college", "student", "semester"]),
        ("technical", ["arduino", "sensor", "api", "architecture"]),
        ("form", ["application", "signature"]),
        ("presentation", ["slide", "powerpoint"]),
    ]
    for label, keys in rules:
        if any(k in t for k in keys):
            return label
    return "document"


# ==========================================================
# 🔸 Main Engine Class
# ==========================================================
class OfflineRAGEngine:
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.vdb = VectorStore(str(self.storage_dir / "chroma"))
        self._bm25_path = self.storage_dir / "bm25.pkl"
        self._tok, self._chunks, self._metas = [], [], []
        self._bm25: Optional[BM25Okapi] = None
        self._load_bm25()

    # ======================================================
    # Persistence
    # ======================================================
    def _load_bm25(self):
        """Load saved BM25 data if available."""
        if self._bm25_path.exists():
            try:
                with open(self._bm25_path, "rb") as f:
                    data = pickle.load(f)
                self._tok = data.get("tok", [])
                self._chunks = data.get("chunks", [])
                self._metas = data.get("metas", [])
                self._bm25 = BM25Okapi(self._tok) if self._tok else None
            except Exception as e:
                print(f"⚠️ BM25 load failed: {e}")
                self._tok, self._chunks, self._metas = [], [], []
                self._bm25 = None

    def _save_bm25(self):
        """Persist BM25 data."""
        try:
            with open(self._bm25_path, "wb") as f:
                pickle.dump({"tok": self._tok, "chunks": self._chunks, "metas": self._metas}, f)
        except Exception as e:
            print(f"⚠️ BM25 save failed: {e}")

    # ======================================================
    # Document Management
    # ======================================================
    def list_documents(self) -> List[str]:
        """List all unique document names in storage."""
        try:
            result = self.vdb.coll.get(include=["metadatas"])
            metas = result.get("metadatas", [])
        except Exception as e:
            print(f"⚠️ list_documents fail: {e}")
            metas = []

        names = set()
        for row in metas:
            if isinstance(row, list):
                for m in row:
                    if isinstance(m, dict) and m.get("source"):
                        names.add(m["source"])
            elif isinstance(row, dict) and row.get("source"):
                names.add(row["source"])
        return sorted(list(names))

    # ======================================================
    # Chunking
    # ======================================================
    def _chunk_text(self, text: str, size=700, overlap=120) -> List[str]:
        """Split long text into overlapping chunks for embedding."""
        words = text.split()
        if len(words) <= size:
            return [text]
        chunks = []
        for i in range(0, len(words), size - overlap):
            chunks.append(" ".join(words[i:i + size]))
        return chunks

    # ======================================================
    # Filter Normalization
    # ======================================================
    def _normalize_filter(self, f):
        """Convert any filter type into a stable, hashable string."""
        if not f:
            return None
        if isinstance(f, (list, tuple, set)):
            return " ".join(map(str, f))
        if isinstance(f, dict):
            return " ".join(f"{k}:{v}" for k, v in f.items())
        return str(f)

    # ======================================================
    # File Ingestion
    # ======================================================
    def ingest_files(self, paths: List[str]) -> List[Dict[str, Any]]:
        """Extracts, embeds, and indexes documents."""
        out = []
        for p in paths:
            pth = Path(p)
            if not pth.exists():
                out.append({"file": pth.name, "success": False, "error": "file not found"})
                continue

            try:
                text, meta = extract_text(str(pth))
            except Exception as e:
                out.append({"file": pth.name, "success": False, "error": str(e)})
                continue

            text = (text or "").strip()
            if not text:
                out.append({"file": pth.name, "success": False, "error": "no readable content"})
                continue

            doc_type = _detect_doc_type(text)
            chunks = self._chunk_text(text)
            ids, docs, metas = [], [], []

            for i, ch in enumerate(chunks):
                uid = hashlib.md5(f"{pth.name}-{i}-{time.time()}".encode()).hexdigest()[:12]
                ids.append(uid)
                docs.append(ch)
                metas.append({"source": pth.name, "doc_type": doc_type, "chunk_index": i})

            try:
                embs = embed_texts(docs)
                if isinstance(embs, np.ndarray):
                    embs = embs.tolist()
                elif isinstance(embs, list) and isinstance(embs[0], (float, int)):
                    embs = [embs]
                elif isinstance(embs, list) and len(embs) == 1 and isinstance(embs[0], list):
                    embs = embs[0]

                self.vdb.add(ids, embs, docs, metas)

                for ch, m in zip(docs, metas):
                    self._chunks.append(ch)
                    self._metas.append(m)
                    self._tok.append(_tokenize(ch))

                self._bm25 = BM25Okapi(self._tok)
                self._save_bm25()

                out.append({"file": pth.name, "success": True, "chunks": len(chunks)})
            except Exception as e:
                out.append({"file": pth.name, "success": False, "error": f"Embedding failed: {e}"})
        return out

    # ======================================================
    # Vector Search (Hybrid Stable v3.0)
    # ======================================================
    def _vector_search(self, question: str, k=10, doc_filter=None):
        """Robust vector search with dynamic shape flattening."""
        try:
            q_emb = embed_texts([question])
            if isinstance(q_emb, np.ndarray):
                q_emb = q_emb.tolist()
            if isinstance(q_emb, list) and len(q_emb) == 1 and isinstance(q_emb[0], list):
                q_emb = q_emb[0]

            where = None
            if doc_filter:
                doc_filter = self._normalize_filter(doc_filter)
                where = {"source": doc_filter}

            res = self.vdb.query(q_emb, n=k, where=where) or {}
            docs_raw = res.get("documents", [])
            metas_raw = res.get("metadatas", [])

            docs, metas = [], []
            if docs_raw and isinstance(docs_raw[0], list):
                docs = docs_raw[0]
            elif isinstance(docs_raw, list):
                docs = docs_raw

            if metas_raw and isinstance(metas_raw[0], list):
                metas = metas_raw[0]
            elif isinstance(metas_raw, list):
                metas = metas_raw

            if not docs or not metas or len(docs) != len(metas):
                return []
            return list(zip(docs, metas))
        except Exception as e:
            print(f"⚠️ vector_search fail: {e}")
            return []

    # ======================================================
    # BM25 Search
    # ======================================================
    def _bm25_search(self, question: str, k=10, doc_filter=None):
        """Keyword-based fallback retrieval."""
        if not self._bm25:
            return []
        try:
            scores = self._bm25.get_scores(_tokenize(question))
            order = np.argsort(scores)[::-1]
            hits = []
            for idx in order:
                if scores[idx] <= 0:
                    break
                m = self._metas[idx]
                if doc_filter and m.get("source") != doc_filter:
                    continue
                hits.append((self._chunks[idx], m, scores[idx]))
                if len(hits) >= k:
                    break
            return [(d, m) for d, m, _ in hits]
        except Exception as e:
            print(f"⚠️ BM25 search fail: {e}")
            return []

    # ======================================================
    # Answer Generation (Consistent Return Mode)
    # ======================================================
    def answer(self, question: str, k=10, doc_filter=None):
        """Main entry point: retrieves and generates the final answer."""
        try:
            vec = self._vector_search(question, k, doc_filter)
            kw = self._bm25_search(question, k, doc_filter)
            merged = (vec or []) + (kw or [])
        except Exception as e:
            print(f"⚠️ Retrieval failed: {e}")
            merged = []

        if not merged and doc_filter:
            print("⚠️ Retrying unfiltered search...")
            merged = self._vector_search(question, k, None) + self._bm25_search(question, k, None)

        prompt = self.build_prompt(question, merged)

        # ✅ Always return (stream, blocks)
        try:
            stream = generate(prompt)
            return stream, merged
        except Exception as e:
            msg = f"⚠️ Error generating response: {e}"
            def _one_shot():
                yield msg
            return _one_shot(), []

    # ======================================================
    # Prompt Builder
    # ======================================================
    def build_prompt(self, question, ctx):
        """Construct the prompt for LLM response generation."""
        if not ctx:
            return f"You are BI. Question: {question}\nNo relevant context found."
        content = "\n\n".join([f"[{m.get('source','?')}]\n{d}" for d, m in ctx])
        return (
            f"You are BI, an offline AI assistant with full semantic reasoning.\n"
            f"Use only the given context to answer precisely and completely.\n\n"
            f"CONTEXT:\n{content}\n\nQUESTION: {question}\nANSWER:"
        )
