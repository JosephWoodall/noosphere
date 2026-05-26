"""
Four-tier memory store for the digital twin agent.

  1. Short-term  — rolling deque of recent (input_features, command) pairs (RAM)
  2. Episodic    — compressed episode summaries with timestamps (SQLite)
  3. Semantic    — FAISS nearest-neighbour index over episode embeddings (disk)
  4. Procedural  — JSON key-value store for learned behavioral rules (disk)

The memory system is deliberately lightweight — no GPU, no SentenceTransformer
at inference time.  Short-term lookup is O(1); semantic search is O(log N).

The primary consumer is the online adaptation loop in digital_twin.py, which
queries short-term memory for recent (x, y) pairs and episodic memory for
"what worked in similar states in the past."
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Short-term memory (RAM ring buffer) ───────────────────────────────────────

@dataclass
class Experience:
    eeg: np.ndarray            # (T, 21) — raw EEG window
    command_pred: np.ndarray   # (6,) — what we predicted
    command_actual: np.ndarray # (6,) — what actually happened (proprioceptive feedback)
    ern_prob: float
    latent: Optional[np.ndarray] = None   # (1, d_model) — encoder output at this step
    timestamp: float = field(default_factory=time.monotonic)


class ShortTermMemory:
    def __init__(self, capacity: int = 512):
        self._buf: deque[Experience] = deque(maxlen=capacity)

    def store(self, exp: Experience):
        self._buf.append(exp)

    def sample(self, n: int) -> list[Experience]:
        if len(self._buf) == 0:
            return []
        n = min(n, len(self._buf))
        indices = np.random.choice(len(self._buf), size=n, replace=False)
        buf_list = list(self._buf)
        return [buf_list[i] for i in indices]

    def recent(self, n: int) -> list[Experience]:
        return list(self._buf)[-n:]

    def __len__(self) -> int:
        return len(self._buf)


# ── Episodic memory (SQLite) ──────────────────────────────────────────────────

class EpisodicMemory:
    """Persistent store of session summaries and key events."""

    def __init__(self, db_path: str = "v2_digital_self_replication/logs/episodes.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                duration_s REAL,
                mean_error REAL,
                n_ern_events INTEGER,
                n_halt_events INTEGER,
                tag TEXT,
                notes TEXT
            )
        """)
        self._conn.commit()

    def log_episode(
        self,
        duration_s: float,
        mean_error: float,
        n_ern: int = 0,
        n_halt: int = 0,
        tag: str = "",
        notes: str = "",
    ):
        self._conn.execute(
            "INSERT INTO episodes (timestamp, duration_s, mean_error, n_ern_events, n_halt_events, tag, notes) "
            "VALUES (?,?,?,?,?,?,?)",
            (time.time(), duration_s, mean_error, n_ern, n_halt, tag, notes),
        )
        self._conn.commit()

    def recent_episodes(self, n: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
        cols = ["id", "timestamp", "duration_s", "mean_error", "n_ern_events",
                "n_halt_events", "tag", "notes"]
        return [dict(zip(cols, row)) for row in rows]

    def close(self):
        self._conn.close()


# ── Procedural memory (JSON rules store) ─────────────────────────────────────

class ProceduralMemory:
    """
    Persistent key-value store for learned behavioral rules.

    Rules are simple strings the agent can inspect or overwrite.
    Example: {"grip_threshold": "0.35", "shoulder_pitch_scale": "0.8"}
    """

    def __init__(self, path: str = "v2_digital_self_replication/logs/procedural.json"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._rules: dict[str, Any] = {}
        if self._path.exists():
            try:
                self._rules = json.loads(self._path.read_text())
            except json.JSONDecodeError:
                logger.warning("ProceduralMemory: corrupt JSON, starting fresh")

    def set(self, key: str, value: Any):
        self._rules[key] = value
        self._path.write_text(json.dumps(self._rules, indent=2))

    def get(self, key: str, default: Any = None) -> Any:
        return self._rules.get(key, default)

    def all_rules(self) -> dict:
        return dict(self._rules)


# ── Semantic memory (FAISS embedding index) ───────────────────────────────────

class SemanticMemory:
    """
    FAISS flat index over mean-pooled EEG embeddings for nearest-neighbour
    episode retrieval.  Falls back gracefully if faiss is not installed.
    """

    def __init__(self, dim: int = 128, path: str = "v2_digital_self_replication/logs/semantic.index"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._dim = dim
        self._index = None
        self._payloads: list[dict] = []
        self._try_load()

    def _try_load(self):
        try:
            import faiss
            idx_path = str(self._path)
            if self._path.exists():
                self._index = faiss.read_index(idx_path)
            else:
                self._index = faiss.IndexFlatL2(self._dim)
        except ImportError:
            logger.info("SemanticMemory: faiss not installed, semantic search disabled")

    def add(self, embedding: np.ndarray, payload: dict):
        if self._index is None:
            return
        import faiss
        vec = embedding.reshape(1, -1).astype(np.float32)
        self._index.add(vec)
        self._payloads.append(payload)
        faiss.write_index(self._index, str(self._path))

    def search(self, query: np.ndarray, k: int = 5) -> list[dict]:
        if self._index is None or self._index.ntotal == 0:
            return []
        import faiss
        vec = query.reshape(1, -1).astype(np.float32)
        k = min(k, self._index.ntotal)
        _, indices = self._index.search(vec, k)
        return [self._payloads[i] for i in indices[0] if i >= 0]


# ── Unified memory facade ─────────────────────────────────────────────────────

class MemoryStore:
    def __init__(
        self,
        capacity: int = 512,
        embedding_dim: int = 128,
        log_dir: str = "v2_digital_self_replication/logs",
    ):
        self.short_term = ShortTermMemory(capacity=capacity)
        self.episodic   = EpisodicMemory(db_path=f"{log_dir}/episodes.db")
        self.procedural = ProceduralMemory(path=f"{log_dir}/procedural.json")
        self.semantic   = SemanticMemory(dim=embedding_dim, path=f"{log_dir}/semantic.index")

    def store_experience(self, exp: Experience):
        self.short_term.store(exp)

    def sample_for_training(self, n: int = 32) -> list[Experience]:
        return self.short_term.sample(n)

    def log_episode(self, **kwargs):
        self.episodic.log_episode(**kwargs)

    def close(self):
        self.episodic.close()
