"""
retriever.py — Vector Database for RAG (ChromaDB)
Robust to nested vectors; compatible with engine.py v6.5 and embedder v2.3
"""
from typing import List, Dict, Any, Optional, Iterable
from pathlib import Path
import chromadb
from chromadb.config import Settings

try:
    import numpy as np
except Exception:
    np = None


def _flatten_1d(vec: Any) -> List[float]:
    """
    Flatten [[[...]]]/[[...]]/np arrays -> plain 1D python list[float].
    Never returns nested lists.
    """
    # numpy array
    if np is not None and isinstance(vec, np.ndarray):
        return vec.astype(float).ravel().tolist()

    # python list/tuple: keep peeling singletons
    if isinstance(vec, (list, tuple)):
        v = vec
        # peel leftmost singletons: [[[x]]] -> [[x]] -> [x] -> x
        while isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        # if still nested (e.g., [[x,y], [z,w]]), flatten fully
        if isinstance(v, (list, tuple)) and any(isinstance(x, (list, tuple)) for x in v):
            flat: List[float] = []
            for item in v:
                if isinstance(item, (list, tuple)):
                    flat.extend([float(xx) for xx in item])
                else:
                    flat.append(float(item))
            return flat
        # now v should be 1D
        return [float(x) for x in v]
    # anything else: best-effort
    try:
        return [float(vec)]
    except Exception:
        # fallback empty vector
        return []


def _ensure_2d_embeddings(embeddings: Any) -> List[List[float]]:
    """
    Accepts:
      - np.ndarray [N, D] or [D]
      - list of vectors, possibly nested
      - single vector
    Returns: List[List[float]] of shape [N, D].
    """
    if np is not None and isinstance(embeddings, np.ndarray):
        if embeddings.ndim == 1:
            return [embeddings.astype(float).ravel().tolist()]
        # ndim >= 2 -> convert each row to list
        return [row.astype(float).ravel().tolist() for row in embeddings]

    # python list/tuple cases
    if isinstance(embeddings, (list, tuple)):
        # if it looks like a single vector (1D), wrap
        if not embeddings or not isinstance(embeddings[0], (list, tuple)):
            return [_flatten_1d(embeddings)]
        # looks like list of vectors -> flatten each row
        return [_flatten_1d(row) for row in embeddings]

    # single vector-like
    return [_flatten_1d(embeddings)]


class VectorStore:
    """Wrapper for ChromaDB vector database — robust to nested vectors."""

    def __init__(self, path: str):
        # Ensure directory exists
        db_path = Path(path)
        db_path.mkdir(parents=True, exist_ok=True)

        # Initialize client with explicit settings
        try:
            self.client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            print(f"✅ ChromaDB initialized at: {db_path}")
        except Exception as e:
            print(f"⚠️ ChromaDB settings error: {e}, trying basic init...")
            self.client = chromadb.PersistentClient(path=str(db_path))
            print(f"✅ ChromaDB initialized (basic mode) at: {db_path}")

        # Get or create collection
        try:
            self.coll = self.client.get_collection("docs")
            doc_count = self.count()
            print(f"✅ Loaded existing collection 'docs' with {doc_count} documents")
        except Exception:
            self.coll = self.client.create_collection(
                "docs", metadata={"hnsw:space": "cosine"}
            )
            print("✅ Created new collection 'docs'")

    def add(
        self,
        ids: List[str],
        embeddings: Any,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Add documents to the collection (embeddings are sanitized)."""
        try:
            emb_2d = _ensure_2d_embeddings(embeddings)
            self.coll.add(
                ids=ids, embeddings=emb_2d, documents=documents, metadatas=metadatas
            )
            print(f"✅ Added {len(ids)} documents to collection")
        except Exception as e:
            print(f"⚠️ Error adding documents: {e}")
            raise

    def query(
        self, query_emb: Any, n: int = 5, where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query the collection (query_emb is sanitized to 1D)."""
        try:
            q = _flatten_1d(query_emb)
            if not q:
                raise ValueError("Empty query vector after flattening")

            kwargs = {"query_embeddings": [q], "n_results": n}
            if where and isinstance(where, dict) and len(where) > 0:
                kwargs["where"] = where

            return self.coll.query(**kwargs)
        except Exception as e:
            print(f"⚠️ Query error: {e}")
            # Return empty results structure that matches expected format
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def count(self) -> int:
        try:
            return self.coll.count()
        except Exception:
            return 0

    def reset(self):
        try:
            self.client.delete_collection("docs")
            self.coll = self.client.create_collection(
                "docs", metadata={"hnsw:space": "cosine"}
            )
            print("✅ Collection reset successfully")
        except Exception as e:
            print(f"⚠️ Error resetting collection: {e}")
