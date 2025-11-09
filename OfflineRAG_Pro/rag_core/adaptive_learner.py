# ==========================================================
# adaptive_learner.py — Feedback-Based Learning Engine
# ==========================================================

import json, os
from collections import defaultdict
from typing import Dict, Any
from datetime import datetime


class AdaptiveLearner:
    """
    Learns from user feedback and adjusts retrieval parameters.
    Works fully offline — stores patterns, scores, and confidence.
    """

    def __init__(self, path: str = "data/learning.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.patterns = defaultdict(list)
        self._load_learning()

    def _load_learning(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.patterns[k] = v
            except Exception:
                pass

    def _save_learning(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.patterns, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save learning file: {e}")

    def _extract_pattern(self, text: str) -> str:
        """Simplify query text to detect repeated patterns."""
        words = text.lower().split()
        key_terms = [w for w in words if len(w) > 3]
        return " ".join(sorted(set(key_terms[:5])))  # core pattern key

    def learn_from_feedback(self, query: str, response: str, feedback: str):
        """
        Store user feedback (positive/negative) for adaptive improvement.
        """
        pattern = self._extract_pattern(query)
        score = 1 if feedback == "positive" else -0.5

        record = {
            "query": query.strip(),
            "response": response.strip(),
            "score": score,
            "timestamp": datetime.now().isoformat()
        }

        self.patterns[pattern].append(record)
        self._save_learning()

    def adaptive_parameters(self, query: str) -> Dict[str, Any]:
        """
        Dynamically tune retrieval/search parameters based on history.
        Example: increase 'k' for well-known topics.
        """
        pattern = self._extract_pattern(query)
        records = self.patterns.get(pattern, [])

        if not records:
            return {"k": 10, "confidence": "medium", "experience": 0}

        avg_score = sum(r["score"] for r in records) / len(records)
        experience = len(records)

        if avg_score > 0.5 and experience >= 5:
            return {"k": 20, "confidence": "high", "experience": experience}
        elif avg_score < 0:
            return {"k": 5, "confidence": "low", "experience": experience}
        else:
            return {"k": 10, "confidence": "medium", "experience": experience}

    def summarize_learning(self) -> Dict[str, Any]:
        """Generate quick summary of learning progress."""
        total = sum(len(v) for v in self.patterns.values())
        pos = sum(1 for v in self.patterns.values() for r in v if r["score"] > 0)
        neg = total - pos
        return {
            "total_feedback": total,
            "positive": pos,
            "negative": neg,
            "accuracy": round((pos / total) * 100, 2) if total > 0 else 0
        }
