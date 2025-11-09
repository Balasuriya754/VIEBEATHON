# ==========================================================
# learning_system.py — Self-Learning Intelligence Core
# ==========================================================

import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np


class ConversationalMemory:
    """
    Learns from user interactions and builds context-aware memory
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Memory files
        self.short_term_path = self.storage_dir / "short_term.json"
        self.long_term_path = self.storage_dir / "long_term.pkl"
        self.preferences_path = self.storage_dir / "user_preferences.json"

        # In-memory structures
        self.short_term: List[Dict[str, Any]] = []
        self.long_term: Dict[str, Any] = defaultdict(list)
        self.preferences: Dict[str, Any] = {}
        self.session_start = datetime.now()

        # Load existing memory
        self._load_memory()

    def _load_memory(self):
        """Load saved memory from disk"""
        try:
            if self.short_term_path.exists():
                with open(self.short_term_path, 'r') as f:
                    self.short_term = json.load(f)
        except Exception:
            self.short_term = []

        try:
            if self.long_term_path.exists():
                with open(self.long_term_path, 'rb') as f:
                    self.long_term = pickle.load(f)
        except Exception:
            self.long_term = defaultdict(list)

        try:
            if self.preferences_path.exists():
                with open(self.preferences_path, 'r') as f:
                    self.preferences = json.load(f)
        except Exception:
            self.preferences = {}

    def _save_memory(self):
        """Persist memory to disk"""
        try:
            with open(self.short_term_path, 'w') as f:
                json.dump(self.short_term[-100:], f, indent=2)  # Keep last 100
        except Exception:
            pass

        try:
            with open(self.long_term_path, 'wb') as f:
                pickle.dump(dict(self.long_term), f)
        except Exception:
            pass

        try:
            with open(self.preferences_path, 'w') as f:
                json.dump(self.preferences, f, indent=2)
        except Exception:
            pass

    def add_interaction(self, query: str, response: str, context: Dict[str, Any] = None):
        """Record a conversation turn"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "context": context or {},
            "session": self.session_start.isoformat()
        }

        # Add to short-term memory
        self.short_term.append(interaction)

        # Extract and store patterns
        self._extract_patterns(query, response)

        # Update long-term memory periodically
        if len(self.short_term) % 10 == 0:
            self._consolidate_to_long_term()
            self._save_memory()

    def _extract_patterns(self, query: str, response: str):
        """Extract patterns from interactions"""
        query_lower = query.lower()

        # Topic extraction (simple keyword-based)
        topics = self._extract_topics(query_lower)
        for topic in topics:
            self.long_term[f"topic:{topic}"].append({
                "query": query,
                "response_preview": response[:200],
                "timestamp": datetime.now().isoformat()
            })

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        # Domain keywords
        domains = {
            'math': ['calculate', 'solve', 'equation', 'derivative', 'integral'],
            'data': ['analyze', 'data', 'statistics', 'mean', 'median'],
            'code': ['code', 'function', 'python', 'script', 'algorithm'],
            'document': ['document', 'summary', 'extract', 'file'],
            'financial': ['finance', 'roi', 'interest', 'investment'],
        }

        detected = []
        for domain, keywords in domains.items():
            if any(kw in text for kw in keywords):
                detected.append(domain)

        return detected

    def _consolidate_to_long_term(self):
        """Move patterns from short-term to long-term memory"""
        # Analyze recent interactions for patterns
        recent = self.short_term[-50:]

        # Count query types
        query_types = Counter()
        for interaction in recent:
            topics = self._extract_topics(interaction['query'].lower())
            for topic in topics:
                query_types[topic] += 1

        # Store aggregated patterns
        if query_types:
            self.long_term['query_type_frequency'].append({
                'timestamp': datetime.now().isoformat(),
                'frequencies': dict(query_types)
            })

    def get_context(self, query: str, n_recent: int = 5) -> List[Dict[str, Any]]:
        """Get relevant context for current query"""
        # Return recent interactions
        return self.short_term[-n_recent:]

    def learn_preference(self, key: str, value: Any):
        """Learn user preference"""
        self.preferences[key] = {
            'value': value,
            'updated': datetime.now().isoformat()
        }
        self._save_memory()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        topics = Counter()
        for interaction in self.short_term:
            extracted = self._extract_topics(interaction['query'].lower())
            topics.update(extracted)

        return {
            'short_term_count': len(self.short_term),
            'long_term_keys': len(self.long_term),
            'preferences': len(self.preferences),
            'top_topics': dict(topics.most_common(5)),
            'session_duration': str(datetime.now() - self.session_start)
        }


class AdaptiveLearner:
    """
    Learns from feedback and improves response quality
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.feedback_path = self.storage_dir / "feedback.pkl"
        self.patterns_path = self.storage_dir / "learned_patterns.pkl"

        # Feedback storage: query_pattern -> [success_score, count]
        self.feedback: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'positive': 0,
            'negative': 0,
            'responses': []
        })

        # Learned patterns
        self.patterns: Dict[str, Any] = {}

        self._load_learning_data()

    def _load_learning_data(self):
        """Load learning data from disk"""
        try:
            if self.feedback_path.exists():
                with open(self.feedback_path, 'rb') as f:
                    self.feedback = pickle.load(f)
        except Exception:
            pass

        try:
            if self.patterns_path.exists():
                with open(self.patterns_path, 'rb') as f:
                    self.patterns = pickle.load(f)
        except Exception:
            pass

    def _save_learning_data(self):
        """Persist learning data"""
        try:
            with open(self.feedback_path, 'wb') as f:
                pickle.dump(dict(self.feedback), f)
        except Exception:
            pass

        try:
            with open(self.patterns_path, 'wb') as f:
                pickle.dump(self.patterns, f)
        except Exception:
            pass

    def record_feedback(self, query: str, response: str, is_positive: bool):
        """Record user feedback on response quality"""
        pattern = self._get_query_pattern(query)

        if is_positive:
            self.feedback[pattern]['positive'] += 1
        else:
            self.feedback[pattern]['negative'] += 1

        self.feedback[pattern]['responses'].append({
            'query': query,
            'response': response[:200],
            'positive': is_positive,
            'timestamp': datetime.now().isoformat()
        })

        # Learn from pattern
        self._update_patterns(pattern)
        self._save_learning_data()

    def _get_query_pattern(self, query: str) -> str:
        """Extract pattern from query"""
        query_lower = query.lower()

        # Pattern keywords
        patterns = [
            ('math_solve', ['solve', 'calculate', 'compute']),
            ('math_derivative', ['derivative', 'differentiate']),
            ('math_integral', ['integral', 'integrate']),
            ('data_analysis', ['analyze', 'statistics', 'mean']),
            ('code_gen', ['code', 'write', 'generate', 'function']),
            ('document_query', ['document', 'file', 'extract']),
            ('explanation', ['explain', 'what is', 'how does']),
        ]

        for pattern_name, keywords in patterns:
            if any(kw in query_lower for kw in keywords):
                return pattern_name

        return 'general'

    def _update_patterns(self, pattern: str):
        """Update learned patterns based on feedback"""
        feedback_data = self.feedback[pattern]
        total = feedback_data['positive'] + feedback_data['negative']

        if total > 5:  # Enough data to learn
            success_rate = feedback_data['positive'] / total

            self.patterns[pattern] = {
                'success_rate': success_rate,
                'total_feedback': total,
                'confidence': 'high' if total > 20 else 'medium' if total > 10 else 'low',
                'updated': datetime.now().isoformat()
            }

    def get_optimal_params(self, query: str) -> Dict[str, Any]:
        """Get optimal retrieval parameters based on learned patterns"""
        pattern = self._get_query_pattern(query)

        # Default parameters
        params = {
            'k': 10,
            'temperature': 0.2,
            'confidence': 'medium'
        }

        if pattern in self.patterns:
            pattern_data = self.patterns[pattern]
            success_rate = pattern_data['success_rate']

            # Adjust based on success rate
            if success_rate > 0.8:
                params['k'] = 15  # Retrieve more for successful patterns
                params['confidence'] = 'high'
            elif success_rate < 0.5:
                params['k'] = 20  # Retrieve more to find better context
                params['temperature'] = 0.3  # More creative responses
                params['confidence'] = 'low'

        return params

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        stats = {
            'patterns_learned': len(self.patterns),
            'total_feedback': sum(
                d['positive'] + d['negative']
                for d in self.feedback.values()
            ),
            'pattern_performance': {}
        }

        for pattern, data in self.patterns.items():
            stats['pattern_performance'][pattern] = {
                'success_rate': f"{data['success_rate'] * 100:.1f}%",
                'confidence': data['confidence']
            }

        return stats


class IntelligenceMetrics:
    """
    Track bot intelligence growth over time
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.storage_dir / "metrics.json"
        self.metrics: Dict[str, List[Any]] = defaultdict(list)

        self._load_metrics()

    def _load_metrics(self):
        """Load metrics from disk"""
        try:
            if self.metrics_path.exists():
                with open(self.metrics_path, 'r') as f:
                    self.metrics = defaultdict(list, json.load(f))
        except Exception:
            pass

    def _save_metrics(self):
        """Save metrics to disk"""
        try:
            with open(self.metrics_path, 'w') as f:
                json.dump(dict(self.metrics), f, indent=2)
        except Exception:
            pass

    def record_query(self, query: str, success: bool, confidence: float, response_time: float):
        """Record a query and its outcome"""
        self.metrics['queries'].append({
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'confidence': confidence,
            'response_time': response_time
        })

        # Calculate rolling metrics
        if len(self.metrics['queries']) % 10 == 0:
            self._calculate_rolling_metrics()
            self._save_metrics()

    def _calculate_rolling_metrics(self):
        """Calculate rolling window metrics"""
        recent = self.metrics['queries'][-50:]

        if recent:
            success_rate = sum(1 for q in recent if q['success']) / len(recent)
            avg_confidence = sum(q['confidence'] for q in recent) / len(recent)
            avg_response_time = sum(q['response_time'] for q in recent) / len(recent)

            self.metrics['rolling_stats'].append({
                'timestamp': datetime.now().isoformat(),
                'success_rate': success_rate,
                'avg_confidence': avg_confidence,
                'avg_response_time': avg_response_time,
                'sample_size': len(recent)
            })

    def get_growth_report(self) -> Dict[str, Any]:
        """Generate intelligence growth report"""
        if not self.metrics['rolling_stats']:
            return {'status': 'insufficient_data'}

        recent_stats = self.metrics['rolling_stats'][-10:]

        if len(recent_stats) < 2:
            return {'status': 'insufficient_history'}

        # Calculate trends
        success_trend = recent_stats[-1]['success_rate'] - recent_stats[0]['success_rate']
        confidence_trend = recent_stats[-1]['avg_confidence'] - recent_stats[0]['avg_confidence']

        return {
            'total_queries': len(self.metrics['queries']),
            'current_success_rate': f"{recent_stats[-1]['success_rate'] * 100:.1f}%",
            'success_trend': 'improving' if success_trend > 0 else 'declining',
            'confidence_trend': 'improving' if confidence_trend > 0 else 'stable',
            'avg_response_time': f"{recent_stats[-1]['avg_response_time']:.2f}s",
            'learning_velocity': f"{abs(success_trend) * 100:.2f}% per 50 queries"
        }


# ==========================================================
# Usage Example
# ==========================================================
if __name__ == "__main__":
    # Initialize learning system
    memory = ConversationalMemory("./learning_storage")
    learner = AdaptiveLearner("./learning_storage")
    metrics = IntelligenceMetrics("./learning_storage")

    # Simulate interaction
    memory.add_interaction(
        "solve x^2 + 5x + 6 = 0",
        "The solutions are x = -2 and x = -3",
        {"type": "math"}
    )

    # Record feedback
    learner.record_feedback(
        "solve x^2 + 5x + 6 = 0",
        "The solutions are x = -2 and x = -3",
        is_positive=True
    )

    # Get stats
    print("Memory Stats:", memory.get_stats())
    print("Learning Stats:", learner.get_learning_stats())
    print("Growth Report:", metrics.get_growth_report())