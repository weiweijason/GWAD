"""Dual-track query screening and cross-session aggregation for GWAD."""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Hashable, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class ScreenResult:
    """Result of screening one query update."""

    suspicious: bool
    delta: Optional[torch.Tensor]
    delta_norm: Optional[float]


class UpdateVarianceScreener:
    """Classify updates as attack-like small moves or benign image switches.

    The first query in a session has no update and is treated as benign. Later
    updates are suspicious when their L2 norm is less than or equal to the
    configured threshold.
    """

    def __init__(self, threshold: float = 0.1):
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = float(threshold)
        self._previous: Dict[Hashable, torch.Tensor] = {}

    def reset(self, session_id: Optional[Hashable] = None) -> None:
        if session_id is None:
            self._previous.clear()
        else:
            self._previous.pop(session_id, None)

    def process(
        self, query: torch.Tensor, session_id: Hashable = "default"
    ) -> ScreenResult:
        current = query.detach()
        previous = self._previous.get(session_id)
        self._previous[session_id] = current.clone()
        if previous is None:
            return ScreenResult(False, None, None)

        if previous.shape != current.shape:
            raise ValueError("consecutive queries must have the same shape")
        delta = current - previous
        delta_norm = float(torch.linalg.vector_norm(delta).item())
        return ScreenResult(delta_norm <= self.threshold, delta, delta_norm)


@dataclass
class _Update:
    session_id: Hashable
    vector: np.ndarray


class CrossSessionGraphAggregator:
    """Merge sessions whose recent update directions are similar."""

    def __init__(
        self, similarity_threshold: float = 0.9, buffer_size: int = 1024
    ):
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if buffer_size < 1:
            raise ValueError("buffer_size must be positive")
        self.similarity_threshold = float(similarity_threshold)
        self.buffer_size = int(buffer_size)
        self._updates: Deque[_Update] = deque(maxlen=buffer_size)
        self._session_queue: Dict[Hashable, int] = {}
        self._queue_sessions: Dict[int, set] = defaultdict(set)
        self._next_queue_id = 0

    @staticmethod
    def _as_vector(delta: torch.Tensor) -> np.ndarray:
        vector = delta.detach().cpu().numpy().reshape(1, -1)
        if not np.isfinite(vector).all():
            raise ValueError("delta must contain only finite values")
        if np.linalg.norm(vector) == 0:
            raise ValueError("delta must be non-zero")
        return vector

    def reset(self) -> None:
        self._updates.clear()
        self._session_queue.clear()
        self._queue_sessions.clear()
        self._next_queue_id = 0

    def find_or_create_queue(
        self, session_id: Hashable, delta: torch.Tensor
    ) -> int:
        vector = self._as_vector(delta)
        matches = [
            update
            for update in reversed(self._updates)
            if update.session_id != session_id
            and cosine_similarity(vector, update.vector)[0, 0]
            >= self.similarity_threshold
        ]

        queue_id = self._session_queue.get(session_id)
        if queue_id is None and matches:
            queue_id = self._session_queue.get(matches[0].session_id)
        if queue_id is None:
            queue_id = self._next_queue_id
            self._next_queue_id += 1

        self._session_queue[session_id] = queue_id
        self._queue_sessions[queue_id].add(session_id)
        self._updates.append(_Update(session_id, vector))
        return queue_id

    def sessions_for_queue(self, queue_id: int) -> frozenset:
        return frozenset(self._queue_sessions.get(queue_id, ()))

    def queue_for_session(self, session_id: Hashable) -> Optional[int]:
        return self._session_queue.get(session_id)


class GWADDualTrackPipeline:
    """Route screened updates into per-queue DS windows.

    ``classify`` is called after a queue reaches ``window_size`` and receives
    the queue's DS values. This keeps the official GWAD classifier reusable.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        similarity_threshold: float = 0.9,
        window_size: int = 256,
        buffer_size: int = 1024,
    ):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.screener = UpdateVarianceScreener(threshold)
        self.aggregator = CrossSessionGraphAggregator(
            similarity_threshold, buffer_size
        )
        self.window_size = int(window_size)
        self._queues: Dict[int, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def reset(self) -> None:
        self.screener.reset()
        self.aggregator.reset()
        self._queues.clear()

    def process_update(
        self,
        query: torch.Tensor,
        session_id: Hashable = "default",
        ds: Optional[float] = None,
    ) -> Tuple[str, Optional[int], Optional[Tuple[float, ...]]]:
        result = self.screener.process(query, session_id)
        if not result.suspicious:
            return "PASS_BENIGN", None, None

        queue_id = self.aggregator.find_or_create_queue(session_id, result.delta)
        if ds is None:
            return "QUEUED", queue_id, None

        queue = self._queues[queue_id]
        queue.append(float(ds))
        if len(queue) < self.window_size:
            return "QUEUED", queue_id, None
        return "READY", queue_id, tuple(queue)
