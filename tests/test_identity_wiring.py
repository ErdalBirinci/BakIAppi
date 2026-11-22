"""Kimlik onarımının canlı ve video hatlarına bağlandığını doğrulayan testler.

``test_identity.py`` yalnızca ``PlayerIdentity`` kurallarını ölçer; buradaki
testler hatların kendisini çalıştırır. YOLO/SigLIP yerine sahte bir
sınıflandırıcı ve betikle verilen tespitler kullanılır; onarımın ürettiği
numara çizime ve anlık duruma kadar ulaşıyor mu, ona bakılır.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import supervision as sv

from app import live, pipeline

FRAME_W, FRAME_H = 64, 48
SLOTS = {
    "A": (2, 8, 14, 40),
    "B": (17, 8, 29, 40),
    "C": (32, 8, 44, 40),
    "D": (47, 8, 59, 40),
}
# BGR: aynı takımdaki iki oyuncu birbirinden, takımlar da birbirinden ayrışsın.
PALETTE = {
    "A": (255, 0, 0),
    "B": (0, 255, 0),
    "C": (0, 0, 255),
    "D": (255, 255, 0),
}
COLOR_TEAM = {
    PALETTE["A"]: 0,
    PALETTE["B"]: 0,
    PALETTE["C"]: 1,
    PALETTE["D"]: 1,
}

# A oyuncusu iki kare görünmez olur, sonra yeni bir tracker ID'si ile döner.
SCRIPT = [
    [("A", 11), ("B", 12), ("C", 21), ("D", 22)],
    [("A", 11), ("B", 12), ("C", 21), ("D", 22)],
    [("B", 12), ("C", 21), ("D", 22)],
    [("A", 98), ("B", 12), ("C", 21), ("D", 22)],
    [("A", 98), ("B", 12), ("C", 21), ("D", 22)],
    [("A", 98)],
]


class FakeClassifier:
    """Kırpımın ortalama rengini gömlem, en yakın palet rengini takım sayar."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def fit(self, crops) -> None:
        pass

    def extract_features(self, crops) -> np.ndarray:
        return np.array(
            [crop.reshape(-1, 3).mean(axis=0) for crop in crops], dtype=np.float32
        )

    def predict_from_features(self, features) -> np.ndarray:
        teams = []
        for vector in features:
            color = min(
                COLOR_TEAM,
                key=lambda candidate: float(
                    np.linalg.norm(np.array(candidate, dtype=np.float32) - vector)
                ),
            )
            teams.append(COLOR_TEAM[color])
        return np.array(teams, dtype=int)


def render(rows) -> np.ndarray:
    frame = np.full((FRAME_H, FRAME_W, 3), 20, dtype=np.uint8)
    for key, _ in rows:
        x1, y1, x2, y2 = SLOTS[key]
        frame[y1:y2, x1:x2] = PALETTE[key]
    return frame


def detect(rows) -> sv.Detections:
    boxes = np.array([SLOTS[key] for key, _ in rows], dtype=np.float32)
    return sv.Detections(
        xyxy=boxes,
        class_id=np.full(len(rows), pipeline.PLAYER_CLASS_ID, dtype=int),
        confidence=np.ones(len(rows), dtype=np.float32),
        tracker_id=np.array([tracker_id for _, tracker_id in rows], dtype=int),
    )


class FakeYOLO:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def to(self, **kwargs):
        return self

    def __call__(self, *args, **kwargs):
        return [object()]


class PassthroughTracker:
    """Tespitler zaten tracker_id taşıdığı için izleyici onları olduğu gibi geçirir."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def update_with_detections(self, detections):
        return detections


class VideoPipelineWiringTest(unittest.TestCase):
    def test_analyze_merges_returning_player_ids(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "source.mp4")
            target = os.path.join(root, "annotated.mp4")
            writer = cv2.VideoWriter(
                source, cv2.VideoWriter_fourcc(*"mp4v"), 5, (FRAME_W, FRAME_H)
            )
            self.assertTrue(writer.isOpened())
            for rows in SCRIPT:
                writer.write(render(rows))
            writer.release()

            script = [detect(rows) for rows in SCRIPT]
            drawn: list[list[int]] = []

            def draw(frame, detections, color_lookup):
                drawn.append([int(tracker_id) for tracker_id in detections.tracker_id])
                return frame

            with mock.patch.object(pipeline, "YOLO", FakeYOLO), mock.patch.object(
                pipeline, "fit_team_classifier", lambda *args, **kwargs: FakeClassifier()
            ), mock.patch.object(
                pipeline.sv, "ByteTrack", PassthroughTracker
            ), mock.patch.object(
                pipeline.sv.Detections,
                "from_ultralytics",
                side_effect=lambda result: script.pop(0),
            ), mock.patch.object(
                pipeline.sv.KeyPoints,
                "from_ultralytics",
                return_value=SimpleNamespace(xy=None, confidence=None),
            ), mock.patch.object(
                pipeline, "annotate_players", side_effect=draw
            ):
                stats = pipeline.analyze(source, target, imgsz=64)

            self.assertEqual(stats["frames_processed"], len(SCRIPT))
            self.assertEqual(stats["id_repairs"], 1)
            self.assertEqual(stats["team_source"], "siglip")
            self.assertEqual(drawn[2], [12, 21, 22])
            self.assertEqual(drawn[3], [11, 12, 21, 22])
            self.assertEqual(drawn[4], [11, 12, 21, 22])
            self.assertEqual(drawn[5], [11])
            self.assertGreater(os.path.getsize(target), 0)


class LiveResolverWiringTest(unittest.TestCase):
    def make_resolver(self) -> live.TeamResolver:
        resolver = live.TeamResolver()
        resolver.classifier = FakeClassifier()
        return resolver

    def test_resolver_reuses_player_id_across_break(self) -> None:
        resolver = self.make_resolver()

        first = detect(SCRIPT[0])
        teams = resolver.resolve(render(SCRIPT[0]), first)

        np.testing.assert_array_equal(teams, [0, 0, 1, 1])
        np.testing.assert_array_equal(first.tracker_id, [11, 12, 21, 22])

        returning = detect(SCRIPT[3])
        teams = resolver.resolve(render(SCRIPT[3]), returning)

        np.testing.assert_array_equal(returning.tracker_id, [11, 12, 21, 22])
        np.testing.assert_array_equal(teams, [0, 0, 1, 1])
        self.assertEqual(resolver.identity.merge_count, 1)
        self.assertEqual(resolver.identity.alias, {98: 11})

    def test_warmup_fit_enrolls_players_before_identity(self) -> None:
        with mock.patch.object(live, "TEAM_FIT_SAMPLES", 4), mock.patch(
            "sports.common.team.TeamClassifier", FakeClassifier
        ):
            resolver = live.TeamResolver()

            first = detect(SCRIPT[0])
            teams = resolver.resolve(render(SCRIPT[0]), first)

            self.assertIsInstance(resolver.classifier, FakeClassifier)
            self.assertFalse(resolver.pending)
            np.testing.assert_array_equal(teams, [0, 0, 1, 1])
            np.testing.assert_array_equal(first.tracker_id, [11, 12, 21, 22])

            returning = detect(SCRIPT[3])
            resolver.resolve(render(SCRIPT[3]), returning)

            np.testing.assert_array_equal(returning.tracker_id, [11, 12, 21, 22])
            self.assertEqual(resolver.identity.merge_count, 1)


class ScriptedCentroid:
    def __init__(self, tracker_ids) -> None:
        self.tracker_ids = tracker_ids

    def update(self, centers, width):
        return np.array(self.tracker_ids, dtype=int)


class LiveProcessWiringTest(unittest.TestCase):
    def test_process_hands_players_to_identity(self) -> None:
        match = live.LiveMatch("m1", 1, "deneme", "http://example.com/stream.m3u8")
        resolver = live.TeamResolver()
        resolver.classifier = FakeClassifier()
        match._resolver = resolver
        player_model, pitch_model = FakeYOLO(), FakeYOLO()

        with mock.patch.object(
            live.sv.Detections,
            "from_ultralytics",
            side_effect=[detect(SCRIPT[0]), detect(SCRIPT[3])],
        ), mock.patch.object(
            live.sv.KeyPoints,
            "from_ultralytics",
            return_value=SimpleNamespace(xy=None, confidence=None),
        ):
            match._process(
                render(SCRIPT[0]),
                player_model,
                pitch_model,
                ScriptedCentroid([11, 12, 21, 22]),
                resolver,
                None,
            )
            match._process(
                render(SCRIPT[3]),
                player_model,
                pitch_model,
                ScriptedCentroid([98, 12, 21, 22]),
                resolver,
                None,
            )

        self.assertEqual(match.samples, 2)
        self.assertEqual(resolver.identity.alias, {98: 11})
        self.assertEqual(resolver.identity.merge_count, 1)


class LiveSnapshotWiringTest(unittest.TestCase):
    def test_snapshot_reports_repairs(self) -> None:
        resolver = live.TeamResolver()
        resolver.classifier = FakeClassifier()
        resolver.resolve(render(SCRIPT[0]), detect(SCRIPT[0]))
        resolver.resolve(render(SCRIPT[3]), detect(SCRIPT[3]))

        match = live.LiveMatch("m1", 1, "deneme", "http://example.com/stream.m3u8")
        match.status = "live"
        match._resolver = resolver

        self.assertEqual(match.snapshot()["id_repairs"], 1)


if __name__ == "__main__":
    unittest.main()
