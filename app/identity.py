"""Oyuncu kimliği: YOLO kırpımlarından görünüm profili ve ID onarımı.

Tracker kimlikleri (ByteTrack / canlı izleyici) kopabildiği için aynı oyuncu
maç içinde birden çok ID alabilir. Bu modül her yeni ID'nin SigLIP görünüm
gömlemini oyuncu profilleriyle karşılaştırır; yeterince benzeyen ve o anda
sahada görünmeyen bir oyuncu varsa yeni ID o oyuncuya bağlanır.

Ham SigLIP gömlemi takım formasını baskın taşır; bu yüzden karşılaştırma
takım ortalaması çıkarılmış vektörlerle yapılır. Eşik gerçek bir maç klibinde
ölçülen ayrıma göre seçildi: aynı oyuncunun yarıları arasındaki benzerlik
p25 ≈ 0.55, farklı oyuncular arasındaki benzerlik p95 ≈ 0.44.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

DEFAULT_THRESHOLD = 0.55
DEFAULT_MARGIN = 0.12
ENABLED = os.environ.get("BAKIAPPI_REID", "1") != "0"


class PlayerIdentity:
    """Oyuncu başına görünüm profili tutar ve kopan ID'leri ona bağlar."""

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        margin: float = DEFAULT_MARGIN,
        enabled: bool = ENABLED,
    ) -> None:
        self.threshold = threshold
        self.margin = margin
        self.enabled = enabled
        self.alias: dict[int, int] = {}
        self.merges: List[dict] = []
        self._sum: dict[int, np.ndarray] = {}
        self._count: dict[int, int] = {}
        self._team: dict[int, int] = {}

    @property
    def merge_count(self) -> int:
        return len(self.merges)

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            return vector
        return vector / norm

    def _profile(self, canonical: int) -> np.ndarray:
        return self._unit(self._sum[canonical] / self._count[canonical])

    def _team_mean(self, team: int) -> Optional[np.ndarray]:
        members = [key for key, value in self._team.items() if value == team and key in self._sum]
        if not members:
            return None
        return self._unit(np.mean([self._profile(key) for key in members], axis=0))

    def _adjusted(self, vector: np.ndarray, team: int) -> np.ndarray:
        unit = self._unit(vector)
        mean = self._team_mean(team)
        return unit if mean is None else self._unit(unit - mean)

    def enroll(self, canonical: int, vector: np.ndarray, team: int) -> None:
        """Oyuncunun profiline yeni bir görünüm örneği ekler."""
        canonical = int(canonical)
        unit = self._unit(vector)
        total = self._sum.get(canonical)
        self._sum[canonical] = unit if total is None else total + unit
        self._count[canonical] = self._count.get(canonical, 0) + 1
        self._team[canonical] = int(team)

    def _best_match(
        self, raw: int, vector: np.ndarray, team: int, visible: set[int]
    ) -> tuple[Optional[int], float, float]:
        """En iyi ve ikinci en iyi adayı döndürür; belirsiz eşleşme birleştirilmez."""
        adjusted = self._adjusted(vector, team)
        best: Optional[int] = None
        best_score = -1.0
        runner_up = -1.0
        for canonical, profile_team in self._team.items():
            if profile_team != team or canonical in visible or canonical == raw:
                continue
            score = float(adjusted @ self._adjusted(self._profile(canonical), team))
            if score > best_score:
                best, best_score, runner_up = canonical, score, best_score
            elif score > runner_up:
                runner_up = score
        return best, best_score, runner_up

    def resolve(
        self,
        tracker_ids,
        embeddings: Optional[List[Optional[np.ndarray]]],
        teams,
    ) -> np.ndarray:
        """Satır başına kararlı oyuncu kimliği döndürür; gerekirse ID birleştirir."""
        ids = np.full(len(tracker_ids), -1, dtype=int)
        if not self.enabled:
            for row, tracker_id in enumerate(tracker_ids):
                if tracker_id is not None:
                    ids[row] = int(tracker_id)
            return ids

        visible: set[int] = set()
        pending: List[tuple[int, int, Optional[np.ndarray], int]] = []

        for row, tracker_id in enumerate(tracker_ids):
            if tracker_id is None:
                continue
            raw = int(tracker_id)
            canonical = self.alias.get(raw, raw)
            vector = None
            if embeddings is not None and row < len(embeddings):
                vector = embeddings[row]
            if canonical in self._sum:
                ids[row] = canonical
                visible.add(canonical)
                if vector is not None:
                    self.enroll(canonical, vector, teams[row])
            else:
                pending.append((row, raw, vector, int(teams[row])))

        for row, raw, vector, team in pending:
            canonical = raw
            if vector is not None:
                match, score, runner_up = self._best_match(raw, vector, team, visible)
                clear = score - runner_up >= self.margin
                if match is not None and score >= self.threshold and clear:
                    canonical = match
                    self.alias[raw] = canonical
                    self.merges.append(
                        {"from": raw, "to": canonical, "score": round(score, 3)}
                    )
                    self.enroll(canonical, vector, team)
                else:
                    self.enroll(raw, vector, team)
            ids[row] = canonical
            visible.add(canonical)

        return ids

    def apply(self, detections, embeddings, teams) -> None:
        """Tespitlerin tracker_id'sini kararlı oyuncu kimliğine çevirir (yerinde).

        Canlı ve video hatları aynı dönüşümü kullansın diye tek noktada tutulur;
        tracker_id'si olmayan satırlar olduğu gibi bırakılır.
        """
        tracker_ids = detections.tracker_id
        if tracker_ids is None:
            return
        canonical = self.resolve(tracker_ids, embeddings, teams)
        for row, tracker_id in enumerate(tracker_ids):
            if tracker_id is not None:
                tracker_ids[row] = canonical[row]
