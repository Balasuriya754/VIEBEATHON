# ==========================================================
# memory.py — Persistent Conversational Memory (Offline BI)
# ==========================================================

import json, os
from datetime import datetime
from typing import List, Dict, Optional


class ConversationalMemory:
    """
    Offline persistent memory for your AI assistant.
    Stores all past user queries, answers, and timestamps.
    """

    def __init__(self, path: str = "data/memory.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data = self._load_memory()

    def _load_memory(self) -> Dict:
        """Load existing memory or create a new one."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"history": []}

    def save_memory(self):
        """Write memory to disk safely."""
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save memory: {e}")

    def add_interaction(self, query: str, response: str):
        """Store a new interaction."""
        record = {
            "query": query.strip(),
            "response": response.strip(),
            "timestamp": datetime.now().isoformat()
        }
        self.data["history"].append(record)
        self.save_memory()

    def recall_similar(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Retrieve similar past interactions using keyword overlap.
        Lightweight offline similarity — no embeddings required.
        """
        q_words = set(query.lower().split())
        scored = []
        for item in self.data.get("history", []):
            past_words = set(item["query"].lower().split())
            score = len(q_words & past_words)
            if score > 0:
                scored.append((score, item))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [x[1] for x in scored[:limit]]

    def clear(self):
        """Erase all stored history."""
        self.data = {"history": []}
        self.save_memory()
