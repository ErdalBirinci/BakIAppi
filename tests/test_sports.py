import io
import unittest
import uuid
from unittest import mock

import numpy as np
import supervision as sv
from PIL import Image

from app import cards, db, heatmap, live, pipeline, server, sports
from tests.test_regressions import TempDataTestCase


class SportProfileTest(unittest.TestCase):
    def test_court_measurements(self) -> None:
        football = sports.get("football")
        basketball = sports.get("basketball")
        volleyball = sports.get("volleyball")
        self.assertEqual((football.court_length_cm, football.court_width_cm), (12000, 7000))
        self.assertEqual((basketball.court_length_cm, basketball.court_width_cm), (2800, 1500))
        self.assertEqual((volleyball.court_length_cm, volleyball.court_width_cm), (1800, 900))
        self.assertEqual(football.period_count, 2)
        self.assertEqual(basketball.period_count, 4)
        self.assertEqual(volleyball.period_count, 5)
        self.assertTrue(football.has_ball)
        self.assertTrue(basketball.has_ball)
        self.assertTrue(volleyball.has_ball)
        self.assertFalse(basketball.has_goalkeeper)
        self.assertFalse(volleyball.has_goalkeeper)
        self.assertEqual(sports.court_image(volleyball).size, (720, 360))

    def test_events_and_points(self) -> None:
        self.assertEqual(sports.get("football").event_points, {"goal": 1})
        self.assertEqual(
            sports.get("basketball").event_points, {"basket": 2, "basket3": 3}
        )
        self.assertIn("assist", sports.get("basketball").event_kinds)
        volleyball = sports.get("volleyball")
        self.assertEqual(
            volleyball.event_points,
            {"ace": 1, "kill": 1, "block": 1, "error": 1},
        )
        self.assertEqual(volleyball.error_kinds, ("error",))
        self.assertEqual(volleyball.period_targets, (25, 25, 25, 25, 15))
        self.assertEqual(volleyball.win_by, 2)

    def test_zones_follow_the_sport(self) -> None:
        self.assertEqual(pipeline.zone_of(12.0, "football"), "jog")
        self.assertEqual(pipeline.zone_of(12.0, "basketball"), "run")
        self.assertEqual(pipeline.zone_of(18.0, "football"), "run")
        self.assertEqual(pipeline.zone_of(18.0, "basketball"), "sprint")
        self.assertEqual(pipeline.zone_of(3.9, "volleyball"), "walk")
        self.assertEqual(pipeline.zone_of(4.0, "volleyball"), "jog")
        self.assertEqual(pipeline.zone_of(8.0, "volleyball"), "run")
        self.assertEqual(pipeline.zone_of(13.0, "volleyball"), "sprint")

    def test_period_guess(self) -> None:
        football = sports.get("football")
        basketball = sports.get("basketball")
        self.assertEqual(sports.period_of(football, 0), 1)
        self.assertEqual(sports.period_of(football, 50 * 60), 2)
        self.assertEqual(sports.period_of(basketball, 20 * 60), 2)
        self.assertEqual(sports.period_of(basketball, 70 * 60), 4)
        volleyball = sports.get("volleyball")
        self.assertEqual(sports.period_of(volleyball, 0), 1)
        self.assertEqual(sports.period_of(volleyball, 30 * 60), 2)
        self.assertEqual(sports.period_of(volleyball, 120 * 60), 5)

    def test_payload_exposes_points_and_calibration(self) -> None:
        rows = {row["key"]: row for row in sports.payload("tr")}
        self.assertEqual(rows["basketball"]["points"], {"basket": 2, "basket3": 3})
        self.assertEqual(rows["basketball"]["calibration"], "manual")
        self.assertEqual(rows["football"]["calibration"], "auto")
        self.assertEqual(rows["basketball"]["court"], {"length_m": 28.0, "width_m": 15.0})
        self.assertEqual(rows["volleyball"]["scoring"], "sets")
        self.assertEqual(rows["volleyball"]["period_targets"], [25, 25, 25, 25, 15])
        self.assertEqual(rows["volleyball"]["error_kinds"], ["error"])
        self.assertEqual(rows["volleyball"]["court"], {"length_m": 18.0, "width_m": 9.0})
        self.assertEqual(rows["volleyball"]["calibration"], "manual")


class CalibrationTest(unittest.TestCase):
    def test_full_court_projection(self) -> None:
        profile = sports.get("basketball")
        matrix = sports.calibration_matrix(
            profile, [[0, 0], [560, 0], [560, 300], [0, 300]], "full"
        )
        centre = sports.project(matrix, np.array([[280.0, 150.0]]))
        self.assertAlmostEqual(float(centre[0][0]), 1400.0, delta=1.0)
        self.assertAlmostEqual(float(centre[0][1]), 750.0, delta=1.0)

    def test_half_court_projection(self) -> None:
        profile = sports.get("basketball")
        corners = sports.court_corners(profile, "half")
        self.assertEqual(corners[-2], (1400.0, 1500.0))
        matrix = sports.calibration_matrix(
            profile, [[0, 0], [560, 0], [560, 300], [0, 300]], "half"
        )
        centre = sports.project(matrix, np.array([[280.0, 150.0]]))
        self.assertAlmostEqual(float(centre[0][0]), 700.0, delta=1.0)

    def test_invalid_points_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sports.calibration_matrix(sports.get("basketball"), [[0, 0], [1, 1]], "full")

    def test_volleyball_full_and_half_court(self) -> None:
        profile = sports.get("volleyball")
        matrix = sports.calibration_matrix(
            profile, [[0, 0], [720, 0], [720, 360], [0, 360]], "full"
        )
        centre = sports.project(matrix, np.array([[360.0, 180.0]]))
        self.assertAlmostEqual(float(centre[0][0]), 900.0, delta=1.0)
        self.assertAlmostEqual(float(centre[0][1]), 450.0, delta=1.0)
        self.assertEqual(sports.court_corners(profile, "half")[-2], (900.0, 900.0))


class BasketballHeatmapTest(unittest.TestCase):
    def test_cells_and_png_size(self) -> None:
        profile = sports.get("basketball")
        bucket: dict[int, int] = {}
        for point in ((100.0, 100.0), (110.0, 120.0), (2700.0, 1400.0)):
            heatmap.accumulate(bucket, point, profile)
        self.assertEqual(sum(bucket.values()), 3)
        self.assertIsNone(heatmap.cell_of((-5, 10), profile))

        data = heatmap.render(bucket, profile)
        self.assertIsNotNone(data)
        image = Image.open(io.BytesIO(data))
        self.assertEqual(image.size, (560, 300))

        box = heatmap.focus_box(bucket, sport=profile)
        self.assertIsNotNone(box)
        left, top, right, bottom = box
        self.assertTrue(0.0 <= left < right <= 1.0)
        self.assertTrue(0.0 <= top < bottom <= 1.0)

    def test_full_court_bucket_does_not_leak_into_basketball_grid(self) -> None:
        football: dict[int, int] = {}
        heatmap.accumulate(football, (11990.0, 6990.0), "football")
        self.assertTrue(football)
        basketball: dict[int, int] = {}
        heatmap.accumulate(basketball, (12000.0, 6990.0), "basketball")
        self.assertEqual(basketball, {})


class VolleyballHeatmapAndSetTest(unittest.TestCase):
    def test_heatmap_size_and_cells(self) -> None:
        profile = sports.get("volleyball")
        bucket: dict[int, int] = {}
        for point in ((10.0, 10.0), (900.0, 450.0), (1790.0, 890.0)):
            heatmap.accumulate(bucket, point, profile)
        self.assertEqual(sum(bucket.values()), 3)
        data = heatmap.render(bucket, profile)
        self.assertIsNotNone(data)
        self.assertEqual(Image.open(io.BytesIO(data)).size, (720, 360))

    def test_set_stops_at_25_and_needs_two_point_lead(self) -> None:
        profile = sports.get("volleyball")
        events = [
            {"id": index, "minute": index, "kind": "ace", "team": 0}
            for index in range(1, 25)
        ] + [
            {"id": 100 + index, "minute": 100 + index, "kind": "kill", "team": 1}
            for index in range(1, 25)
        ]
        events += [
            {"id": 200, "minute": 200, "kind": "block", "team": 0},
        ]
        state = sports.volleyball_sets(profile, events)
        self.assertEqual(state["sets"], [])
        self.assertEqual(state["current"], [25, 24])
        state = sports.volleyball_sets(
            profile, events + [{"id": 201, "minute": 201, "kind": "ace", "team": 0}]
        )
        self.assertEqual(state["sets"], [{"set": 1, "a": 26, "b": 24, "winner": "A"}])
        self.assertEqual(state["wins"], [1, 0])
        self.assertEqual(state["current"], [0, 0])

    def test_error_event_credits_the_opponent_and_fifth_set_is_15(self) -> None:
        profile = sports.get("volleyball")
        error_state = sports.volleyball_sets(
            profile, [{"id": 1, "minute": 1, "kind": "error", "team": 1}]
        )
        # Hata B oyuncusunun; sayı otomatik olarak A'ya yazılır.
        self.assertEqual(error_state["current"], [1, 0])

        # İlk dört set 2-2; beşinci sette B 15-13 ile seti kazanır.
        events = []
        for set_no in range(4):
            team = set_no % 2
            events += [
                {
                    "id": set_no * 100 + i,
                    "minute": set_no * 100 + i,
                    "kind": "ace",
                    "team": team,
                }
                for i in range(1, 26)
            ]
        events += [
            {"id": 500 + i, "minute": 500 + i * 2, "kind": "kill", "team": 1}
            for i in range(1, 16)
        ]
        events += [
            {"id": 600 + i, "minute": 501 + i * 2, "kind": "ace", "team": 0}
            for i in range(1, 14)
        ]
        state = sports.volleyball_sets(profile, events)
        self.assertEqual(len(state["sets"]), 5)
        self.assertEqual(state["sets"][-1], {"set": 5, "a": 13, "b": 15, "winner": "B"})
        self.assertEqual(state["wins"], [2, 3])
        self.assertTrue(state["complete"])


class RadarAndMotionTest(unittest.TestCase):
    def test_basketball_radar_draws(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        points = np.array([[1400.0, 750.0], [500.0, 400.0]])
        colors = np.array([0, 1])
        out = pipeline.draw_radar(frame, points, colors, "basketball")
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(int(out.sum()), 0)

    def test_volleyball_court_and_motion(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        points = np.array([[900.0, 450.0], [500.0, 400.0]])
        colors = np.array([0, 1])
        out = pipeline.draw_radar(frame, points, colors, "volleyball")
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(int(out.sum()), 0)

        class Motion:
            interval_s = 1.0

            def distance_step(self, tracker_id):
                return 250.0

        entry = {"distance_m": 0.0, "top_speed_kmh": 0.0}
        self.assertTrue(pipeline.record_motion(entry, Motion(), 1, sport="volleyball"))
        self.assertAlmostEqual(entry["zones"]["run"], 2.5, places=2)
        self.assertEqual(entry["zones"]["jog"], 0.0)

    def test_record_motion_uses_basketball_thresholds(self) -> None:
        class Motion:
            interval_s = 1.0

            def distance_step(self, tracker_id):
                return 333.3

        entry = {"distance_m": 0.0, "top_speed_kmh": 0.0}
        self.assertTrue(pipeline.record_motion(entry, Motion(), 1, sport="basketball"))
        self.assertAlmostEqual(entry["zones"]["run"], 3.333, places=2)
        self.assertEqual(entry["zones"]["jog"], 0.0)

    def test_card_meta_uses_sport_speed_scale(self) -> None:
        meta = cards.build_meta(
            title="x",
            date="d",
            group="g",
            match_id="abcdef",
            elapsed_s=60,
            sport="basketball",
        )
        self.assertEqual(meta["sport"], "basketball")
        self.assertLess(meta["max_speed_kmh"], 30.0)

        volleyball = cards.build_meta(
            title="x",
            date="d",
            group="g",
            match_id="fedcba",
            elapsed_s=60,
            sport="volleyball",
            score_text="Set 3 · 18–16 · Setler 1–0",
        )
        self.assertEqual(volleyball["sport"], "volleyball")
        self.assertEqual(volleyball["max_speed_kmh"], 26.0)
        self.assertEqual(volleyball["score_text"], "Set 3 · 18–16 · Setler 1–0")


if __name__ == "__main__":
    unittest.main()


class VolleyballCourtTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = sports.get("volleyball")
        self.matrix = sports.calibration_matrix(
            self.profile, [[0, 0], [720, 0], [720, 360], [0, 360]], "full"
        )

    def _players(self, bottoms):
        boxes = np.array(
            [[x - 10, y - 30, x + 10, y, ] for x, y in bottoms], dtype=float
        )
        return sv.Detections(
            xyxy=boxes,
            confidence=np.ones(len(boxes)),
            class_id=np.zeros(len(boxes), dtype=int),
            tracker_id=np.arange(len(boxes), dtype=int),
        )

    def test_filter_keeps_servers_and_drops_sideline_bench(self) -> None:
        players = self._players([(360, 180), (10, 180), (360, 385)])
        kept = pipeline._on_court(players, self.matrix, self.profile)
        self.assertEqual(kept.tracker_id.tolist(), [0, 1])

    def test_teams_follow_court_sides_without_siglip(self) -> None:
        players = self._players([(180, 180), (540, 180)])
        teams, _ = pipeline.resolve_player_teams(
            np.zeros((360, 720, 3), dtype=np.uint8),
            players,
            None,
            {},
            self.profile,
            self.matrix,
        )
        self.assertEqual(teams.tolist(), [0, 1])

        resolver = live.TeamResolver(use_classifier=False)
        live_teams = resolver.resolve(
            np.zeros((360, 720, 3), dtype=np.uint8), players, self.matrix, 1800.0
        )
        self.assertEqual(live_teams.tolist(), [0, 1])


class VolleyballServerTest(TempDataTestCase):
    """Sunucu olay kaydı: hata rakibe sayı yazar, rapor set özetini taşır."""

    def test_error_credits_opponent_and_report_has_sets(self) -> None:
        match_id = uuid.uuid4().hex[:12]
        db.create_match(
            match_id, self.group_id, "Voleybol test", "local.mp4", sport="volleyball"
        )
        match = db.get_match(match_id, self.group_id)
        group = {"id": self.group_id, "name": "Test", "username": "test"}
        players = [{"tracker_id": 1, "team": 0}]
        with mock.patch.object(server, "current_group", lambda request: group), mock.patch.object(
            server, "_match_players", lambda current: players
        ):
            error = server.add_match_event(
                match_id, server.EventPayload(tracker_id=1, kind="error", minute=3), None
            )["event"]
            # Olay oyuncunun takımını taşır; sayıyı rakip alır.
            self.assertEqual(error["team"], 0)
            for index in range(25):
                db.add_event(match_id, 1, 0, "ace", float(4 + index))

        with mock.patch.object(
            server,
            "_analysis_of",
            lambda current: {"oyuncular": [], "takimlar": {}, "tarih": ""},
        ):
            text = server._report_text(match, "tr")
        self.assertIn("Setler:", text)
        self.assertIn("25–1", text)
        self.assertIn("Hatalar (rakip sayı):", text)

    def test_get_match_exposes_the_same_set_state(self) -> None:
        match_id = uuid.uuid4().hex[:12]
        db.create_match(
            match_id, self.group_id, "Voleybol set API", "local.mp4", sport="volleyball"
        )
        db.add_event(match_id, 1, 0, "ace", 1.0)
        group = {"id": self.group_id, "name": "Test", "username": "test"}
        with mock.patch.object(server, "current_group", lambda request: group):
            payload = server.get_match(match_id, None)
        self.assertEqual(payload["sets"]["wins"], [0, 0])
        self.assertEqual(payload["sets"]["current"], [1, 0])
