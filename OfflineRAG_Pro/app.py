
# ==========================================================
# bi_ai_v6.4.4.py — BI AI Neural Intelligence Console (All-in-One)
# ==========================================================
"""
BI AI v6.4.4 — Fully Offline, Self-Learning, Multimodal AI Console
- RAG + Local LLM (Ollama) + FAISS + Knowledge Graph
- Math, Data, Forecast, Vision, Voice (5s recording)
- Real-time token streaming, memory sync, image context
- High-contrast, glassmorphic, cinematic UI

Fixes in this build:
  ✅ MathEngine import: robust fallback loader for rag_core/math_engine.py
  ✅ Unhashable doc_filter: normalized + safe fallback (no crashes, no double warnings)
  ✅ LaTeX rendering for math results in chat
  ✅ Minor stability + logs
"""


from __future__ import annotations

import sys, os
base_dir = os.path.dirname(__file__)
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, "rag_core"))
import os, sys, time, tempfile, importlib.util
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================================
# Safe imports with fallbacks
# ==========================================================
def _try_import(path: str, name: str):
    """
    Try importing a module by dotted path; return (module or None, error or None).
    """
    try:
        mod = __import__(path, fromlist=[name])
        return getattr(mod, name), None
    except Exception as e:
        return None, e

# Core engine
OfflineRAGEngine, _err_engine = _try_import('rag_core.engine', 'OfflineRAGEngine')
if _err_engine:
    print(f"⚠️ OfflineRAGEngine unavailable: {_err_engine}")
    OfflineRAGEngine = None

# MathEngine — special robust loader so your existing file always works
def _load_math_engine_robust():
    MathEngine, err = _try_import('rag_core.math_engine', 'MathEngine')
    if MathEngine:
        return MathEngine
    # Fallback: locate rag_core/math_engine.py relative to this script and import directly
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path(os.getcwd()).resolve()
    candidates = [
        base_dir / 'rag_core' / 'math_engine.py',
        base_dir.parent / 'rag_core' / 'math_engine.py',
    ]
    for f in candidates:
        if f.exists():
            spec = importlib.util.spec_from_file_location("math_engine_fallback", str(f))
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, 'MathEngine'):
                    print("✅ Loaded MathEngine via direct file import fallback.")
                    return getattr(mod, 'MathEngine')
            except Exception as e:
                print(f"⚠️ MathEngine fallback import failed from {f}: {e}")
                continue
    print(f"⚠️ MathEngine unavailable: {err}")
    return None

MathEngine = _load_math_engine_robust()

# Learning system
try:
    from rag_core.learning_system import ConversationalMemory, AdaptiveLearner
except Exception as e:
    print(f"⚠️ Learning system unavailable: {e}")
    ConversationalMemory = None
    AdaptiveLearner = None

# Data analysis engine
try:
    from rag_core.data_analysis import DataAnalysisEngine
except Exception as e:
    print(f"⚠️ DataAnalysisEngine unavailable: {e}")
    DataAnalysisEngine = None

# AI intelligence helpers
try:
    from rag_core.ai_intelligence import (
        SentimentIntentDetector,
        NeuralSelfLearner,
        KnowledgeGraphBuilder,
        TimeSeriesForecaster,
    )
except Exception as e:
    print(f"⚠️ AI intelligence modules unavailable: {e}")
    SentimentIntentDetector = None
    NeuralSelfLearner = None
    KnowledgeGraphBuilder = None
    TimeSeriesForecaster = None

# Local LLM (Ollama)
try:
    from rag_core.local_llm import generate as local_llm_generate
except Exception as e:
    print(f"⚠️ Local LLM unavailable: {e}")
    local_llm_generate = None

# Voice & Vision (optional)
VOICE_AVAILABLE = False
VISION_AVAILABLE = False
try:
    import sounddevice as sd
    import soundfile as sf
    import whisper
    VOICE_AVAILABLE = True
except Exception:
    pass

try:
    from rag_core.vision import analyze_image
    VISION_AVAILABLE = True
except Exception:
    pass


# ==========================================================
# Utils
# ==========================================================
def _s(x) -> str:
    """Safe string — prevents unhashable/list/dict leaks."""
    if x is None:
        return ""
    if isinstance(x, (str, bytes)):
        return x if isinstance(x, str) else x.decode("utf-8", "ignore")
    if isinstance(x, (list, tuple, set)):
        return " ".join(map(_s, x))
    if isinstance(x, dict):
        return " ".join(f"{_s(k)}:{_s(v)}" for k, v in x.items())
    try:
        if isinstance(x, np.ndarray):
            return " ".join(map(_s, x.tolist()))
    except Exception:
        pass
    return str(x)


# ==========================================================
# Voice Recording (5s live)
# ==========================================================
class VoiceRecorder:
    def __init__(self, model_size="base"):
        if not VOICE_AVAILABLE:
            raise ImportError("Voice unavailable")
        self.model = whisper.load_model(model_size)

    def record_and_transcribe(self, duration=5, sample_rate=16000) -> str:
        st.info(f"🎤 Recording for {duration} seconds... Speak now!")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        st.success("✅ Recording complete! Transcribing...")
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, recording, sample_rate)
        result = self.model.transcribe(temp_file.name, fp16=False, language="en")
        text = result.get("text", "").strip()
        try:
            os.unlink(temp_file.name)
        except:
            pass
        return text


# ==========================================================
# Core Orchestrator
# ==========================================================
class BIIntelligenceCore:
    def __init__(self):
        self.engine = OfflineRAGEngine("./rag_storage") if OfflineRAGEngine else None
        self.math = MathEngine() if MathEngine else None
        self.data = DataAnalysisEngine() if DataAnalysisEngine else None
        self.learner = AdaptiveLearner("./learning_storage") if AdaptiveLearner else None
        self.memory = ConversationalMemory("./learning_storage") if ConversationalMemory else None
        self.sentiment = SentimentIntentDetector() if SentimentIntentDetector else None
        self.neural = NeuralSelfLearner("./learning_storage") if NeuralSelfLearner else None
        self.graph = KnowledgeGraphBuilder() if KnowledgeGraphBuilder else None
        self.forecaster = TimeSeriesForecaster() if TimeSeriesForecaster else None

    # --- FIX: normalize doc_filter; never pass unhashable values to engine ---
    def _normalize_doc_filter(self, df):
        """Return a hashable doc_filter or None if not usable."""
        if not df:
            return None
        if isinstance(df, str):
            s = df.strip()
            return s if s else None
        try:
            from collections.abc import Iterable
            if isinstance(df, Iterable) and not isinstance(df, (bytes, bytearray, dict)):
                return tuple(str(x).strip() for x in df if str(x).strip())
        except Exception:
            pass
        return str(df).strip() or None

    def detect_type(self, q: str) -> str:
        x = q.lower()
        if any(k in x for k in ("derivative", "integral", "solve", "equation", "matrix", "statistics", "mean", "median", "std")):
            return "math"
        if any(k in x for k in ("forecast", "predict", "future", "time series", "trend")):
            return "forecast"
        if any(k in x for k in ("analyze", "analysis", "correlation", "dataset", "data")):
            return "data"
        if any(k in x for k in ("graph", "relation", "knowledge graph", "network")):
            return "graph"
        if any(k in x for k in ("write", "story", "essay", "creative", "idea")):
            return "llm"
        return "document"

    # ======================================================
    # Safe CSV
    # ======================================================
    def safe_read_csv(self, path: Path) -> Optional[pd.DataFrame]:
        for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        raise Exception("Failed to read CSV — unsupported encoding")

    # ======================================================
    # Forecast adapter
    # ======================================================
    def forecast_df(self, df, date_col, y_col, periods=14) -> Dict[str, Any]:
        try:
            if not self.forecaster:
                return {"success": False, "error": "Forecaster unavailable"}
            return self.forecaster.forecast(df, date_col, y_col, periods)
        except Exception as e:
            return {"success": False, "error": _s(e)}

    # ======================================================
    # Documents listing
    # ======================================================
    def list_documents(self) -> List[str]:
        try:
            if self.engine and hasattr(self.engine, "list_documents"):
                docs = self.engine.list_documents() or []
                return [str(d) for d in docs]
        except Exception:
            pass
        return []

    # ======================================================
    # Learning hooks (best-effort, non-blocking)
    # ======================================================
    def _learn(self, query: str, response: str, qtype: str = "dialogue"):
        q, r = _s(query), _s(response)
        if not q or not r:
            return
        try:
            if self.neural:
                self.neural.update_vector(q, r)
        except Exception as e:
            print(f"[neural] {e}")
        try:
            if self.memory:
                self.memory.add_interaction(q, r, {"type": qtype})
        except Exception as e:
            print(f"[memory] {e}")
        try:
            if self.graph:
                self.graph.add_document(f"{q}. {r}", source="dialogue")
        except Exception as e:
            print(f"[graph] {e}")

    # ======================================================
    # Streaming Response
    # ======================================================
    def stream_response(self, query: str, focus_doc=None, image_context=None):
        start = time.time()
        query = _s(query)
        focus_doc = _s(focus_doc) if focus_doc else None
        image_context = _s(image_context) if image_context else None

        qtype = self.detect_type(query)
        si = self.sentiment.analyze(query) if self.sentiment else {"intent": "unknown", "sentiment": "neutral"}
        meta = {
            "type": qtype,
            "intent": si.get("intent", "unknown"),
            "sentiment": si.get("sentiment", "neutral"),
            "timestamp": datetime.now().isoformat(),
            "response_time": None
        }

        # Attach image context if provided (to bias local LLM/RAG)
        full_q = query
        if image_context:
            full_q += f"\n\n[Image Context]\n{image_context}"

        safe_focus = None
        if focus_doc and focus_doc.strip() and focus_doc.lower() != "all":
            safe_focus = str(focus_doc).strip()
            full_q = f"[Focus: {safe_focus}]\n{full_q}"

        try:
            # === Math Mode ===
            if qtype == "math":
                if not self.math:
                    yield "⚠️ MathEngine unavailable. Ensure rag_core/math_engine.py exists."
                    meta["response_time"] = round(time.time() - start, 2)
                    return meta
                res = self.math.solve_query(full_q)
                if isinstance(res, dict) and res.get("success", False):
                    # Render LaTeX if available
                    parts = []
                    if "latex" in res:
                        parts.append("$$" + res["latex"] + "$$")
                    # Add other fields (skip huge blobs)
                    for k, v in res.items():
                        if k in ("success", "latex"):
                            continue
                        parts.append(f"**{_s(k).replace('_',' ').title()}:** {_s(v)}")
                    txt = "\n\n".join(parts)
                else:
                    err = res.get("error") if isinstance(res, dict) else "Unable to compute"
                    txt = f"❌ Error: {_s(err)}"
                yield txt
                self._learn(query, txt, qtype="math")
                meta["response_time"] = round(time.time() - start, 2)
                return meta

            # === Forecast Mode ===
            if qtype == "forecast":
                yield "📈 Upload a time-series CSV in the sidebar → Forecast section and click **Run Forecast**."
                meta["response_time"] = round(time.time() - start, 2)
                return meta

            # === Data Mode ===
            if qtype == "data":
                yield "📊 Upload CSV/Excel in the sidebar → **Data Analysis** → Analyze Data."
                meta["response_time"] = round(time.time() - start, 2)
                return meta

            # === Graph Mode ===
            if qtype == "graph":
                if not self.graph:
                    yield "⚠️ Knowledge Graph module unavailable."
                else:
                    g = self.graph.visualize_summary()
                    yield f"🌐 Knowledge Graph: **{g.get('nodes','?')}** nodes • **{g.get('edges','?')}** edges"
                meta["response_time"] = round(time.time() - start, 2)
                return meta

            # === LLM Mode (Creative) ===
            if qtype == "llm":
                if not local_llm_generate:
                    yield "⚠️ Local LLM unavailable. Ensure Ollama is running (`ollama serve`) and model pulled."
                    meta["response_time"] = round(time.time() - start, 2)
                    return meta
                buf = []
                for token in local_llm_generate(full_q, stream=True):
                    t = _s(token)
                    buf.append(t)
                    yield t
                resp = "".join(buf)
                if resp and resp[-1] not in ".!?":
                    yield "."
                    resp += "."
                self._learn(query, resp, qtype="llm")
                meta["response_time"] = round(time.time() - start, 2)
                return meta

            # === Document RAG Mode ===
            if not self.engine:
                yield "⚠️ OfflineRAG engine unavailable."
                meta["response_time"] = round(time.time() - start, 2)
                return meta

            # Normalize doc_filter and call engine safely
            use_filter = self._normalize_doc_filter(safe_focus)
            try:
                if use_filter is None:
                    stream, _ = self.engine.answer(full_q, k=10)
                else:
                    try:
                        stream, _ = self.engine.answer(full_q, k=10, doc_filter=use_filter)
                    except Exception as e:
                        if "unhashable" in str(e).lower() or "list" in str(e).lower():
                            print(f"⚠️ Doc filter not hashable ({use_filter!r}); retrying without filter.")
                            stream, _ = self.engine.answer(full_q, k=10)
                        else:
                            raise
            except TypeError:
                # Some engine builds may not accept doc_filter kwarg at all
                stream, _ = self.engine.answer(full_q, k=10)

            buf = []
            for t in stream:
                t = _s(t)
                buf.append(t)
                yield t
            resp = "".join(buf)
            self._learn(query, resp, "document")
            meta["response_time"] = round(time.time() - start, 2)
            return meta

        except Exception as e:
            error_msg = f"⚠️ Error: {_s(e)}"
            print(f"[ERROR] {error_msg}")
            yield error_msg
            meta["response_time"] = round(time.time() - start, 2)
            return meta


# ==========================================================
# Streamlit UI
# ==========================================================
st.set_page_config(
    page_title="BI AI v6.4.4 — Neural Console",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root {
  --bg: #0E141B;
  --panel: #141C25;
  --panel2:#0B1117;
  --accent: #2EE3C8;
  --ink: #E9F1FA;
  --ink-dim: #A9B6C6;
  --chip: #18222F;
  --line: #1E2733;
}
html, body, .stApp { background: var(--bg); color: var(--ink); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #10171E, #0B1117); border-right: 1px solid var(--line); }
.stButton>button, .stDownloadButton>button {
  background: var(--chip); border: 1px solid #2b3545; border-radius: 12px; color: var(--ink); transition: all .2s; font-weight: 500;
}
.stButton>button:hover { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(46,227,200,.18) inset; }
.upload-box { background: var(--panel); border: 1px dashed #2b3442; padding: .8rem; border-radius: 14px; text-align: center; color: var(--ink-dim);}
.topbar { background: var(--panel2); padding: .8rem 1rem; border-radius: 14px; border: 1px solid var(--line); display:flex; align-items:center; gap:.6rem; }
.badge { background:#13202b; color:#b5c7de; padding:.15rem .55rem; border-radius:999px; font-size:.72rem; border:1px solid #203041; }
.chat-wrap { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; min-height: 68vh; max-height: 68vh; overflow-y: auto; padding: .6rem; }
.msg { padding: .9rem 1.1rem; border-radius: 14px; margin:.35rem 0; line-height: 1.5; }
.msg.user { background: var(--chip); color: var(--ink); border: 1px solid #2b3545; }
.msg.assistant { background: #0f1722; color: #ecf3ff; border: 1px solid #223049; }
.msg .meta { font-size: .72rem; color: var(--ink-dim); margin-bottom: .25rem; }
.bubble { white-space: pre-wrap; word-wrap: break-word;}
.focus-badge { background: var(--accent); color:#000; padding:.3rem .8rem; border-radius:20px; font-weight:700; display:inline-block; }
.glow { animation: pulse 1.4s infinite; }
@keyframes pulse { 0%{opacity:.4;} 50%{opacity:1;} 100%{opacity:.4;} }
.icon-btn { width: 48px !important; height: 48px !important; padding: 0 !important; font-size: 24px !important; display: flex !important; align-items: center !important; justify-content: center !important;}
hr { border-top: 1px solid var(--line); }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# Session State Init
# ==========================================================
if "core" not in st.session_state:
    with st.spinner("🧠 Initializing BI AI Neural Core..."):
        st.session_state.core = BIIntelligenceCore()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "focus_doc" not in st.session_state:
    st.session_state.focus_doc = None
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()
if "show_image_panel" not in st.session_state:
    st.session_state.show_image_panel = False
if "voice_query_pending" not in st.session_state:
    st.session_state.voice_query_pending = None
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "data_file_name" not in st.session_state:
    st.session_state.data_file_name = None

core = st.session_state.core

# ==========================================================
# Sidebar
# ==========================================================
with st.sidebar:
    st.title("🧠 BI AI v6.4.4")
    st.caption("Offline • Private • Self-Learning")

    st.markdown("---")
    st.subheader("📚 Knowledge Base")
    files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=["pdf", "txt", "csv", "docx", "xlsx"],
        key="kb_uploads"
    )
    st.markdown("<div class='upload-box'>📂 Drag & drop files here</div>", unsafe_allow_html=True)

    if core.engine and files:
        for f in files:
            if f.name in st.session_state.ingested_files:
                st.caption(f"✅ {f.name} (indexed)")
                continue
            p = Path(tempfile.gettempdir()) / f.name
            p.write_bytes(f.getbuffer())
            ph = st.empty()
            ph.info(f"🔄 Indexing **{f.name}** ...")
            try:
                with st.spinner("Building vectors..."):
                    core.engine.ingest_files([str(p)])
                st.session_state.ingested_files.add(f.name)
                ph.success(f"✅ {f.name} indexed")
                time.sleep(0.3)
                ph.empty()
            except Exception as e:
                ph.error(f"❌ {f.name}: {str(e)[:100]}")

    st.markdown("---")
    st.subheader("🎯 Document Focus")
    docs = core.list_documents() if core.engine else []
    if docs:
        chosen = st.selectbox("Focus on document:", ["All Documents"] + docs, key="focus_select")
        st.session_state.focus_doc = None if chosen == "All Documents" else chosen
        if st.session_state.focus_doc:
            st.info(f"🔍 Focused: {st.session_state.focus_doc}")
    else:
        st.caption("No documents indexed yet")

    st.markdown("---")
    st.subheader("📊 Data Analysis")
    df_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], key="data_file")

    if df_file:
        if st.session_state.data_file_name != df_file.name:
            tmp = Path(tempfile.gettempdir()) / df_file.name
            tmp.write_bytes(df_file.getbuffer())
            try:
                if tmp.suffix.lower() == ".csv":
                    st.session_state.current_df = core.safe_read_csv(tmp)
                else:
                    st.session_state.current_df = pd.read_excel(tmp)
                st.session_state.data_file_name = df_file.name
            except Exception as e:
                st.error(f"Load error: {str(e)[:100]}")
                st.session_state.current_df = None

        if st.session_state.current_df is not None:
            df = st.session_state.current_df
            st.success(f"✅ {df.shape[0]} rows × {df.shape[1]} cols")

            with st.expander("Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)

            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                col = st.selectbox("📈 Visualize Column:", num_cols, key="viz_col")
                if st.button("Generate Chart", use_container_width=True):
                    try:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(df[col], linewidth=2)
                        ax.set_title(f"{col} Trend")
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Chart error: {str(e)[:100]}")

    st.markdown("---")
    st.subheader("📈 Time Series Forecast")
    fc = st.file_uploader("Upload time-series CSV", type=["csv"], key="forecast_file")

    if fc:
        try:
            tmp = Path(tempfile.gettempdir()) / fc.name
            tmp.write_bytes(fc.getbuffer())
            df_fc = core.safe_read_csv(tmp)

            date_cols = [c for c in df_fc.columns if "date" in c.lower() or "time" in c.lower()]
            num_cols = df_fc.select_dtypes(include=[np.number]).columns.tolist()

            if date_cols and num_cols:
                dc = st.selectbox("Date Column:", date_cols, key="dc")
                yc = st.selectbox("Target Column:", num_cols, key="yc")
                h = st.slider("Forecast Horizon (days):", 5, 60, 14, key="hzn")

                if st.button("🚀 Run Forecast", use_container_width=True):
                    with st.spinner("Forecasting..."):
                        res = core.forecast_df(df_fc, dc, yc, h)

                    if res.get("success"):
                        st.success("✅ Forecast Complete")
                        st.json(res)
                        try:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(pd.to_datetime(df_fc[dc]), df_fc[yc], linewidth=2, label='Historical')
                            ax.set_title(f"{yc} Forecast")
                            ax.legend()
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            st.warning(f"Visualization error: {str(e)[:100]}")
                    else:
                        st.error(f"❌ {res.get('error', 'Forecast failed')}")
            else:
                st.warning("Need a date/time column and a numeric target column")
        except Exception as e:
            st.error(f"File error: {str(e)[:100]}")

# ==========================================================
# Top bar + Focus badge
# ==========================================================
col_top1, col_top2 = st.columns([10, 2])
with col_top1:
    st.markdown(
        "<div class='topbar'>"
        "<span class='badge'>🔒 Offline</span>"
        "<span class='badge'>🔍 FAISS</span>"
        "<span class='badge'>🦙 Ollama</span>"
        "<span class='badge'>🎤 Voice</span>"
        "<span class='badge'>👁️ Vision</span>"
        "</div>",
        unsafe_allow_html=True
    )
with col_top2:
    st.write("")

st.markdown("<hr>", unsafe_allow_html=True)

if st.session_state.focus_doc:
    st.markdown(
        f"<span class='focus-badge glow'>🎯 Focus: {st.session_state.focus_doc}</span>",
        unsafe_allow_html=True
    )

# ==========================================================
# Chat Area
# ==========================================================
st.markdown("<div class='chat-wrap'>", unsafe_allow_html=True)

for m in st.session_state.messages:
    role = m["role"]
    meta = m.get("meta", {})
    qtype = (meta.get("query_type") or "").upper()
    sentiment = meta.get("sentiment", "")
    resp_time = meta.get("response_time", "")

    time_badge = f" • ⚡{resp_time}s" if resp_time else ""
    content = _s(m["content"])

    # If content contains LaTeX blocks $$...$$, Streamlit will render them when using st.markdown(..., unsafe_allow_html=False)
    # We keep HTML wrapper for bubble; LaTeX is still supported with st.markdown below.
    st.markdown(
        f"<div class='msg {'assistant' if role == 'assistant' else 'user'}'>"
        f"<div class='meta'>{role.upper()} • {qtype} • {sentiment}{time_badge}</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    # Render message text separately to allow math
    st.markdown(content)

# Image Analysis Panel
if st.session_state.show_image_panel:
    st.markdown("---")
    st.markdown("### 🖼️ Image Analysis")
    img = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="img_upl", label_visibility="collapsed")

    if img:
        st.image(img, use_container_width=True)
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            if VISION_AVAILABLE:
                tmp = Path(tempfile.gettempdir()) / img.name
                tmp.write_bytes(img.getbuffer())
                try:
                    with st.spinner("Analyzing image..."):
                        res = analyze_image(str(tmp), do_ocr=True, do_detect=True, do_llava=False)

                    parts = []
                    if res.get("ocr_text") and res["ocr_text"] != "[No readable text]":
                        t = res["ocr_text"]
                        parts.append("**📝 Text:**\n" + (t[:1000] + ("..." if len(t) > 1000 else "")))

                    if res.get("detections"):
                        labels = [d.get("label") for d in res["detections"]
                                  if isinstance(d, dict) and d.get("label")]
                        if labels:
                            parts.append("**🎯 Objects:** " + ", ".join(sorted(set(labels))))

                    msg = "\n\n".join(parts) if parts else "No content detected in image."

                except Exception as e:
                    msg = f"⚠️ Analysis error: {str(e)[:200]}"

                st.info(msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg,
                    "meta": {"query_type": "image"}
                })
                st.session_state.show_image_panel = False
                st.rerun()
            else:
                st.error("❌ Vision module unavailable. Check rag_core.vision installation.")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# Handle pending voice query
# ==========================================================
if st.session_state.voice_query_pending:
    user_q = st.session_state.voice_query_pending
    st.session_state.voice_query_pending = None

    # Get recent image context if available
    img_ctx = None
    for m in reversed(st.session_state.messages[-3:]):
        if m.get("meta", {}).get("query_type") == "image":
            img_ctx = m["content"]
            break

    # Analyze query
    q_meta = core.sentiment.analyze(user_q) if core.sentiment else {"sentiment": "neutral"}
    q_type = core.detect_type(user_q)

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_q,
        "meta": {"query_type": q_type, "sentiment": q_meta.get("sentiment", "neutral")}
    })

    # Stream response
    placeholder = st.empty()
    stream_text = ""
    last_meta: Dict[str, Any] = {}

    for token in core.stream_response(user_q, focus_doc=st.session_state.focus_doc, image_context=img_ctx):
        if isinstance(token, dict):
            last_meta = token
            continue
        stream_text += _s(token)
        placeholder.markdown(
            f"<div class='msg assistant'><div class='meta'>ASSISTANT • {q_type.upper()} • Generating...</div></div>",
            unsafe_allow_html=True
        )
        st.markdown(stream_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": stream_text,
        "meta": {"query_type": q_type, "response_time": last_meta.get("response_time")}
    })
    st.rerun()

# First-run welcome
if not st.session_state.messages:
    st.info(
        "👋 **Welcome to BI AI v6.4.4**\n\n"
        "Click 🎤 to record a 5-second voice query, 🖼️ for image analysis, or type below. "
        "BI AI learns locally and runs fully offline with complete privacy.\n\n"
        "Tip: Try a math query like *derivative of sin^2(x) with respect to x*."
    )

# ==========================================================
# Input bar with icon buttons
# ==========================================================
c1, c2, c3 = st.columns([1, 10, 1])

with c1:
    if st.button("🖼️", help="Image analysis", use_container_width=True, key="img_btn"):
        st.session_state.show_image_panel = True
        st.rerun()

with c2:
    with st.form("chat_form", clear_on_submit=True):
        user_q = st.text_input(
            "Ask BI AI… (Press Enter to send)",
            label_visibility="collapsed",
            key="input_box",
            placeholder="What would you like to know?"
        )
        submitted = st.form_submit_button("Send", use_container_width=True)

with c3:
    if st.button("🎤", help="Voice query (5 sec)", use_container_width=True, key="mic_btn"):
        if VOICE_AVAILABLE:
            try:
                recorder = VoiceRecorder()
                text = recorder.record_and_transcribe(duration=5)
                if text:
                    st.success(f"✅ Transcribed: {text}")
                    st.session_state.voice_query_pending = text
                    st.rerun()
                else:
                    st.warning("⚠️ No speech detected. Please try again.")
            except Exception as e:
                st.error(f"❌ Voice error: {str(e)[:200]}")
        else:
            st.error(
                "❌ Voice not available.\n\n"
                "Install: `pip install sounddevice soundfile openai-whisper`"
            )

# ==========================================================
# Handle text input
# ==========================================================
if submitted and user_q:
    # Get recent image context if available
    img_ctx = None
    for m in reversed(st.session_state.messages[-3:]):
        if m.get("meta", {}).get("query_type") == "image":
            img_ctx = m["content"]
            break

    # Analyze query
    q_meta = core.sentiment.analyze(user_q) if core.sentiment else {"sentiment": "neutral"}
    q_type = core.detect_type(user_q)

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_q,
        "meta": {"query_type": q_type, "sentiment": q_meta.get("sentiment", "neutral")}
    })

    # Stream response
    placeholder = st.empty()
    stream_text = ""
    last_meta: Dict[str, Any] = {}

    for token in core.stream_response(user_q, focus_doc=st.session_state.focus_doc, image_context=img_ctx):
        if isinstance(token, dict):
            last_meta = token
            continue
        stream_text += _s(token)
        placeholder.markdown(
            f"<div class='msg assistant'><div class='meta'>ASSISTANT • {q_type.upper()} • Generating...</div></div>",
            unsafe_allow_html=True
        )
        st.markdown(stream_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": stream_text,
        "meta": {
            "query_type": q_type,
            "response_time": last_meta.get("response_time"),
            "sentiment": q_meta.get("sentiment")
        }
    })
    st.rerun()

# ==========================================================
# Footer info
# ==========================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption(f"📝 **Messages:** {len(st.session_state.messages)}")
with col_f2:
    st.caption(f"📚 **Indexed Docs:** {len(st.session_state.ingested_files)}")
with col_f3:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.show_image_panel = False
        st.rerun()

