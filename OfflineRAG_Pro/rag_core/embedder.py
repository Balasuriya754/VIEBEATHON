# ==========================================================
# embedder.py — Offline Embedding Engine (v2.4 FINAL)
# ==========================================================
"""
Features:
✅ 100% Offline — No Internet Required
✅ Works with FAISS + BM25 + Chroma
✅ Direct PyTorch Transformer Load (MiniLM)
✅ Manual cache (safe — no lru_cache list issue)
✅ Hash fallback for total offline safety
✅ Always returns List[List[float]] (FAISS-safe)
✅ Compatible with engine v6.5
"""
# ==========================================================

import numpy as np
import warnings
from pathlib import Path
from typing import List, Union

warnings.filterwarnings("ignore")

_model = None
_tokenizer = None
_model_type = None  # transformer / sentence_transformer / hash
_embed_cache = {}   # manual cache instead of lru_cache


# ==========================================================
# 🔧 Transformer Loader (Direct Offline)
# ==========================================================
def _load_transformer_model_direct():
    """Load MiniLM model from local cache."""
    global _model, _tokenizer, _model_type
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        possible_paths = [
            Path.home() / ".cache" / "huggingface" / "hub" /
            "models--sentence-transformers--all-MiniLM-L6-v2",
            Path("./rag_core/models/MiniLM"),
            Path("./models/MiniLM"),
        ]

        model_path = None
        for path in possible_paths:
            if path.exists():
                snapshots = list((path / "snapshots").iterdir()) if (path / "snapshots").exists() else []
                model_path = snapshots[0] if snapshots else path
                break

        if not model_path or not model_path.exists():
            print("⚠️ No local MiniLM model found — fallback to hash mode.")
            return False

        print(f"✅ Loading MiniLM transformer from: {model_path}")
        _tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        _model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        _model.eval()
        _model_type = "transformer"
        return True

    except Exception as e:
        print(f"⚠️ Transformer model load failed: {e}")
        return False


# ==========================================================
# 🔢 Mean Pooling for Embeddings
# ==========================================================
def _mean_pooling(model_output, attention_mask):
    import torch
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# ==========================================================
# 💠 Hash-Based Fallback Embedding
# ==========================================================
def _hash_embed(text: str, dim: int = 384) -> np.ndarray:
    """Fast hash embedding (fallback)."""
    import hashlib
    vec = np.zeros(dim, dtype=np.float32)
    for i, token in enumerate(text.lower().split()):
        idx1 = abs(hash(token)) % dim
        vec[idx1] += 1.0
        idx2 = int(hashlib.md5(token.encode()).hexdigest(), 16) % dim
        vec[idx2] += 0.8
        vec[(idx1 + idx2 + i) % dim] += 0.5
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ==========================================================
# 🧩 Model Loader
# ==========================================================
def _load_model():
    global _model, _model_type
    if _model_type:
        return _model_type

    print("🔄 Initializing Embedding Engine...")

    # Strategy 1: Direct Transformers
    if _load_transformer_model_direct():
        return "transformer"

    # Strategy 2: SentenceTransformer (local)
    try:
        from sentence_transformers import SentenceTransformer
        for path in [
            Path.home() / ".cache" / "huggingface" / "hub" /
            "models--sentence-transformers--all-MiniLM-L6-v2",
            Path("./rag_core/models/MiniLM"),
            Path("./models/MiniLM"),
        ]:
            if path.exists():
                try:
                    print(f"✅ Loading SentenceTransformer from: {path}")
                    _model = SentenceTransformer(str(path))
                    _model_type = "sentence_transformer"
                    return "sentence_transformer"
                except Exception:
                    continue
    except Exception:
        pass

    print("⚠️ No ML model found — switching to hash embedding mode.")
    _model_type = "hash"
    return "hash"


# ==========================================================
# 🧠 Embedding Generator (Safe Cached)
# ==========================================================
def embed_texts(texts: Union[str, List[str]]) -> List[List[float]]:
    """Generate embeddings — always returns List[List[float]]."""
    import numpy as np

    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return [[]]

    # Make cache key hashable (avoid unhashable 'list')
    key = tuple(sorted(map(str, texts)))
    if key in _embed_cache:
        return _embed_cache[key]

    model_type = _load_model()
    try:
        if model_type == "transformer":
            import torch
            encoded = _tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                output = _model(**encoded)
                pooled = _mean_pooling(output, encoded["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                result = pooled.cpu().numpy().tolist()

        elif model_type == "sentence_transformer":
            embs = _model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            result = np.array(embs, dtype=float).tolist()

        else:
            # Hash fallback
            result = np.array([_hash_embed(t) for t in texts]).tolist()

    except Exception as e:
        print(f"⚠️ Embedding generation failed: {e}")
        result = np.array([_hash_embed(t) for t in texts]).tolist()

    _embed_cache[key] = result
    return result


# ==========================================================
# 🧾 Info Utility
# ==========================================================
def get_embedding_info() -> dict:
    model_type = _load_model()
    return {
        "type": model_type,
        "dimension": 384,
        "offline": True,
        "description": {
            "transformer": "Offline transformer MiniLM model",
            "sentence_transformer": "SentenceTransformer MiniLM model",
            "hash": "Offline hash-based embedding fallback",
        }.get(model_type, "Unknown backend"),
    }


if __name__ == "__main__":
    print("🧠 Testing Offline Embedding Engine...")
    vecs = embed_texts(["Hello world", "Offline RAG test"])
    print("✅ Shape:", np.array(vecs).shape)
    print("ℹ️", get_embedding_info())
