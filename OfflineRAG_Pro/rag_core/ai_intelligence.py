# ==========================================================
# ai_intelligence.py — Advanced AI Intelligence Engines (v2.3 FINAL)
# ==========================================================
"""
Includes (all offline-safe with fallbacks):
- Sentiment & Intent Detection (VADER -> tiny lexicon)
- Knowledge Graph Builder (spaCy -> simple NP parser)
- Time-Series Forecasting (Prophet -> ARIMA -> rolling-mean naive)
- Neural Self-Learning (FAISS -> NumPy cosine) with safe persistence
"""

from __future__ import annotations
import os
import pickle
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

# ---------- Optional deps (lazy, no downloads) ----------
_spacy = None
_nlp = None
_Prophet = None
_ARIMA = None
_SIA = None
_nltk_ok = False
_faiss = None

def _lazy_imports():
    global _spacy, _nlp, _Prophet, _ARIMA, _SIA, _nltk_ok, _faiss
    try:
        import spacy as _sp
        _spacy = _sp
        try:
            _nlp = _spacy.load("en_core_web_sm")
        except Exception:
            # offline-safe: blank english model
            _nlp = _spacy.blank("en")
    except Exception:
        _spacy = None
        _nlp = None

    try:
        from prophet import Prophet as _P
        _Prophet = _P
    except Exception:
        _Prophet = None

    try:
        from statsmodels.tsa.arima.model import ARIMA as _A
        _ARIMA = _A
    except Exception:
        _ARIMA = None

    # Sentiment
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer as _S
        _SIA = _S()
        _nltk_ok = True
    except Exception:
        _SIA = None
        _nltk_ok = False

    # FAISS
    try:
        import faiss as _f
        _faiss = _f
    except Exception:
        _faiss = None

_lazy_imports()

# Small, static lexicons for fallback sentiment
_POS_WORDS = set("good great awesome excellent nice love like happy success win improve improved improvement".split())
_NEG_WORDS = set("bad poor terrible awful hate dislike sad fail failure worse worst bug error issue".split())

# ==========================================================
# SENTIMENT & INTENT DETECTOR
# ==========================================================
class SentimentIntentDetector:
    def __init__(self):
        self.intent_keywords = {
            "query": ["what", "how", "show", "find", "list"],
            "forecast": ["predict", "forecast", "future", "trend"],
            "analysis": ["analyze", "trend", "statistic", "data", "correlation"],
            "greeting": ["hello", "hi", "hey"],
            "exit": ["bye", "exit", "quit"],
        }
        self.sia = _SIA  # may be None

    def _fallback_sentiment(self, text: str) -> str:
        toks = [t.strip(".,!?;:()[]{}\"'").lower() for t in text.split()]
        pos = sum(1 for t in toks if t in _POS_WORDS)
        neg = sum(1 for t in toks if t in _NEG_WORDS)
        if pos - neg > 1: return "positive"
        if neg - pos > 1: return "negative"
        return "neutral"

    def analyze(self, text: str) -> Dict[str, Any]:
        tl = (text or "").lower()
        if self.sia:
            try:
                s = self.sia.polarity_scores(tl)
                sentiment = "neutral"
                if s["compound"] > 0.2: sentiment = "positive"
                elif s["compound"] < -0.2: sentiment = "negative"
            except Exception:
                sentiment = self._fallback_sentiment(tl)
        else:
            sentiment = self._fallback_sentiment(tl)

        intent = "unknown"
        for k, kws in self.intent_keywords.items():
            if any(kw in tl for kw in kws):
                intent = k
                break

        return {"sentiment": sentiment, "intent": intent}

# ==========================================================
# KNOWLEDGE GRAPH BUILDER
# ==========================================================
try:
    import networkx as nx
except Exception:
    nx = None

class KnowledgeGraphBuilder:
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.graph = nx.Graph() if nx else None
        self.spacy_ok = _nlp is not None

    def _simple_spans(self, text: str) -> List[str]:
        # Ultra-simple NP extractor: sequences of TitleCase tokens or ALLCAPS words
        spans = []
        tokens = text.split()
        cur = []
        for t in tokens:
            if (len(t) > 1 and (t[0].isupper() or t.isupper())) and t.isalpha():
                cur.append(t.strip(",.;:!?()[]{}"))
            else:
                if len(cur) > 0:
                    spans.append(" ".join(cur))
                    cur = []
        if cur:
            spans.append(" ".join(cur))
        # Deduplicate and trim
        uniq = []
        seen = set()
        for s in spans:
            s2 = s.strip()
            if len(s2) >= 2 and s2 not in seen:
                uniq.append(s2)
                seen.add(s2)
        return uniq[:50]

    def add_document(self, doc_text: str, source: str = "Unknown"):
        if not self.graph:
            return
        text = (doc_text or "")
        ents: List[str] = []
        if self.spacy_ok:
            try:
                doc = _nlp(text[:100000])  # cap for speed
                ents = list({ent.text for ent in doc.ents if ent.text.strip()})
            except Exception:
                ents = self._simple_spans(text)
        else:
            ents = self._simple_spans(text)

        # Build pairwise edges
        for i, e1 in enumerate(ents):
            for j, e2 in enumerate(ents):
                if i < j:
                    self.graph.add_edge(e1, e2, source=source)

    def visualize_summary(self) -> Dict[str, Any]:
        if not self.graph:
            return {"nodes": 0, "edges": 0, "top_entities": []}
        degrees = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)
        top = [n for n, _ in degrees[:10]]
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "top_entities": top,
        }

    def get_related_entities(self, entity: str) -> List[str]:
        if not self.graph or entity not in self.graph:
            return []
        return list(self.graph.neighbors(entity))

# ==========================================================
# TIME-SERIES FORECASTING
# ==========================================================
class TimeSeriesForecaster:
    def __init__(self):
        self.model = None

    def _naive_roll(self, df: pd.DataFrame, date_col: str, target_col: str, periods: int) -> Dict[str, Any]:
        # Simple rolling mean as fallback
        ser = df.sort_values(date_col)[target_col].astype(float)
        window = min(7, max(3, len(ser)//10 or 3))
        mean = float(pd.Series(ser).rolling(window, min_periods=1).mean().iloc[-1])
        preds = [mean for _ in range(periods)]
        return {
            "success": True,
            "method": "naive_rolling_mean",
            "forecast": {str(i+1): float(v) for i, v in enumerate(preds)}
        }

    def forecast(self, df: pd.DataFrame, date_col: str, target_col: str, periods: int = 10) -> Dict[str, Any]:
        try:
            mdf = df.copy()
            mdf[date_col] = pd.to_datetime(mdf[date_col], errors="coerce")
            mdf = mdf.dropna(subset=[date_col, target_col])
        except Exception as e:
            return {"success": False, "error": f"Bad dataframe: {e}"}

        if _Prophet:
            try:
                tmp = mdf.rename(columns={date_col: "ds", target_col: "y"})
                m = _Prophet()
                m.fit(tmp)
                future = m.make_future_dataframe(periods=periods)
                fc = m.predict(future)
                tail = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
                return {
                    "success": True,
                    "method": "Prophet",
                    "forecast": tail.to_dict(orient="records"),
                }
            except Exception as e:
                pass

        if _ARIMA:
            try:
                ser = mdf.set_index(date_col)[target_col].astype(float)
                model = _ARIMA(ser, order=(2, 1, 2)).fit()
                pred = model.forecast(steps=periods)
                return {
                    "success": True,
                    "method": "ARIMA",
                    "forecast": {str(i+1): float(v) for i, v in enumerate(pred)},
                }
            except Exception:
                pass

        # Fallback naive
        return self._naive_roll(mdf, date_col, target_col, periods)

# ==========================================================
# NEURAL SELF-LEARNING — FAISS + NumPy Fallback (Stable v2.3)
# ==========================================================
def _clean_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (list, tuple, set)):
        return " ".join(map(str, x))
    if isinstance(x, dict):
        return " ".join(f"{k}:{v}" for k, v in x.items())
    if isinstance(x, np.ndarray):
        return " ".join(map(str, x.tolist()))
    return str(x)

def _norm_vec(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    n = np.linalg.norm(v) + 1e-10
    return v / n

def _embed_one(text: str) -> np.ndarray:
    # Import here to avoid circulars
    try:
        from embedder import embed_texts
    except Exception:
        from .embedder import embed_texts
    emb = embed_texts([text])
    if isinstance(emb, np.ndarray):
        vec = emb[0]
    else:
        vec = np.array(emb[0], dtype=np.float32)
    return _norm_vec(vec)

class NeuralSelfLearner:
    """
    Persistent self-learning memory.
    - FAISS if available (Inner Product ≈ Cosine)
    - Otherwise NumPy matrix with cosine similarity
    """
    def __init__(self, storage_dir: str = "./learning_storage"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.dimension = 384  # must match embedder
        self.meta_path = os.path.join(storage_dir, "memory_meta.pkl")

        self.faiss_ok = _faiss is not None
        if self.faiss_ok:
            self.index_path = os.path.join(storage_dir, "faiss_index.bin")
            self._faiss_index = None
            self._text_map: Dict[int, Tuple[str, str]] = {}
            self._load_faiss()
        else:
            self.npy_vecs_path = os.path.join(storage_dir, "vectors.npy")
            self._mat: Optional[np.ndarray] = None  # [N, D]
            self._pairs: List[Tuple[str, str]] = []
            self._load_numpy()

    # ----------------- FAISS backend -----------------
    def _load_faiss(self):
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
                self._faiss_index = _faiss.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    self._text_map = pickle.load(f)
                print(f"✅ FAISS memory loaded ({len(self._text_map)} entries)")
            else:
                self._faiss_index = _faiss.IndexFlatIP(self.dimension)
                self._text_map = {}
                print("🆕 New FAISS index initialized")
        except Exception as e:
            print(f"⚠️ FAISS load failed: {e}. Falling back to NumPy.")
            self.faiss_ok = False
            self._load_numpy()

    def _save_faiss(self):
        try:
            _faiss.write_index(self._faiss_index, self.index_path)
            with open(self.meta_path, "wb") as f:
                pickle.dump(self._text_map, f)
        except Exception as e:
            print(f"⚠️ Error saving FAISS memory: {e}")

    # ----------------- NumPy backend -----------------
    def _load_numpy(self):
        try:
            if os.path.exists(self.npy_vecs_path) and os.path.exists(self.meta_path):
                self._mat = np.load(self.npy_vecs_path)
                with open(self.meta_path, "rb") as f:
                    self._pairs = pickle.load(f)
                print(f"✅ NumPy memory loaded ({len(self._pairs)} entries)")
            else:
                self._mat = np.zeros((0, self.dimension), dtype=np.float32)
                self._pairs = []
                print("🆕 New NumPy memory initialized")
        except Exception as e:
            print(f"⚠️ NumPy load failed: {e}. Starting fresh.")
            self._mat = np.zeros((0, self.dimension), dtype=np.float32)
            self._pairs = []

    def _save_numpy(self):
        try:
            np.save(self.npy_vecs_path, self._mat)
            with open(self.meta_path, "wb") as f:
                pickle.dump(self._pairs, f)
        except Exception as e:
            print(f"⚠️ Error saving NumPy memory: {e}")

    # ----------------- Public API -----------------
    def update_vector(self, query: Any, response: Any):
        q = _clean_text(query)
        r = _clean_text(response)
        if not q or not r:
            return

        # Skip duplicates (prefix signature)
        sig = (q[:80], r[:80])

        if self.faiss_ok:
            if any((qq[:80], rr[:80]) == sig for _, (qq, rr) in self._text_map.items()):
                return
            vec = _norm_vec((_embed_one(q) + _embed_one(r)) / 2.0)
            vec = np.expand_dims(vec, axis=0)
            idx = len(self._text_map)
            self._faiss_index.add(vec)
            self._text_map[idx] = (q, r)
            self._save_faiss()
        else:
            if any((qq[:80], rr[:80]) == sig for (qq, rr) in self._pairs):
                return
            vec = _norm_vec((_embed_one(q) + _embed_one(r)) / 2.0)
            self._mat = np.vstack([self._mat, vec.reshape(1, -1)])
            self._pairs.append((q, r))
            self._save_numpy()

    def find_similar(self, query: Any, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.faiss_ok:
            if not self._text_map or self._faiss_index is None:
                return []
            qv = _embed_one(_clean_text(query)).reshape(1, -1)
            scores, idxs = self._faiss_index.search(qv, top_k)
            out = []
            for s, i in zip(scores[0], idxs[0]):
                if int(i) in self._text_map:
                    out.append((self._text_map[int(i)][0], float(s)))
            return out
        else:
            if self._mat is None or self._mat.shape[0] == 0:
                return []
            qv = _embed_one(_clean_text(query)).reshape(1, -1)
            sims = (qv @ self._mat.T)[0]
            order = np.argsort(sims)[::-1][:top_k]
            return [(self._pairs[i][0], float(sims[i])) for i in order]

    def sync_from_conversational_memory(self, memory_obj, limit: Optional[int] = None) -> int:
        """Import past Q&A pairs from memory object that exposes .interactions list of dicts."""
        if not hasattr(memory_obj, "interactions"):
            return 0
        added = 0
        if self.faiss_ok:
            existing = {(q[:80], r[:80]) for _, (q, r) in self._text_map.items()}
        else:
            existing = {(q[:80], r[:80]) for (q, r) in self._pairs}

        interactions = memory_obj.interactions[-limit:] if (limit and limit > 0) else memory_obj.interactions
        for it in interactions:
            q = _clean_text(it.get("query"))
            r = _clean_text(it.get("response"))
            if not q or not r:
                continue
            sig = (q[:80], r[:80])
            if sig in existing:
                continue
            self.update_vector(q, r)
            existing.add(sig)
            added += 1
        return added

    def clear_memory(self):
        if self.faiss_ok:
            self._faiss_index = _faiss.IndexFlatIP(self.dimension)
            self._text_map = {}
            self._save_faiss()
        else:
            self._mat = np.zeros((0, self.dimension), dtype=np.float32)
            self._pairs = []
            self._save_numpy()

    def stats(self) -> Dict[str, Any]:
        if self.faiss_ok:
            return {
                "entries": len(self._text_map),
                "dimension": self.dimension,
                "backend": "FAISS (Inner Product ≈ Cosine)",
                "index_path": os.path.join(self.storage_dir, "faiss_index.bin"),
                "meta_path": self.meta_path,
            }
        return {
            "entries": int(self._mat.shape[0]) if isinstance(self._mat, np.ndarray) else 0,
            "dimension": self.dimension,
            "backend": "NumPy cosine",
            "vectors_path": getattr(self, "npy_vecs_path", ""),
            "meta_path": self.meta_path,
        }
