"""Kalıcı regresyon testleri.

Kilitlenen davranışlar:

1. Remap uç noktası isimleri yanlış imzayla kaydettiği için 500 veriyordu.
2. Maç biter bitmez silinirse, arka plandaki kart üretimi silinmiş maç için
   dosya ve boş klasör bırakıyordu.
3. Remap'in birleştirme modu kaynağı hedefte toplamalı ve kaynak kaydı silmeli.

Testler geçici bir veri kökü kullanır; gerçek ``app/data`` içeriğine dokunmaz.
Çalıştırma komutu README'de.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

from app import cards, db, heatmap, live, server


class TempDataTestCase(unittest.TestCase):
    """Her testi kendi geçici veri kökünde çalıştırır."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        root = Path(self._temp.name) / "data"
        root.mkdir()
        self.root = root

        for module, name, value in (
            (db, "DATA_DIR", root),
            (db, "DB_PATH", root / "test.db"),
            (db, "ANALYSIS_DIR", root / "analysis"),
            (db, "VIDEO_DIR", root / "videos"),
            (server, "REPORTS_DIR", root / "reports"),
            (cards, "CARD_ROOT", root / "cards"),
            (heatmap, "CACHE_DIR", root / "heatmaps"),
        ):
            patcher = mock.patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        # sqlite3 bağlam yöneticisi bağlantıyı kapatmaz; Windows'ta geçici
        # veritabanı dosyası açık kalmasın diye commit/rollback + close yap.
        real_connect = db.connect

        @contextlib.contextmanager
        def closing_connect():
            connection = real_connect()
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

        connector = mock.patch.object(db, "connect", closing_connect)
        connector.start()
        self.addCleanup(connector.stop)

        for folder in (
            db.ANALYSIS_DIR,
            db.VIDEO_DIR,
            server.REPORTS_DIR,
            cards.CARD_ROOT,
            heatmap.CACHE_DIR,
        ):
            folder.mkdir(parents=True, exist_ok=True)
        db.init()
        server._ANALYSIS_CACHE.clear()

        self.group_id = db.create_group("Regresyon", f"reg_{uuid.uuid4().hex[:8]}", "pass123")

    def create_match(self) -> str:
        match_id = uuid.uuid4().hex[:12]
        db.create_match(match_id, self.group_id, "Regresyon maçı", "local.mp4")
        return match_id


class RemapRegressionTest(TempDataTestCase):
    """Kimlik düzeltmesi 500 vermemeli; isim, olay ve önbellekleri taşımalı."""

    def setUp(self) -> None:
        super().setUp()
        self.match_id = self.create_match()
        players = [
            {
                "tracker_id": 1,
                "team": 0,
                "distance_m": 100.0,
                "top_speed_kmh": 20.0,
                "avg_speed_kmh": 5.0,
                "samples": 10,
                "sprints": 3,
                "high_speed_m": 40.0,
                "zones": {"run": 50.0, "sprint": 10.0},
            },
            {
                "tracker_id": 2,
                "team": 0,
                "distance_m": 200.0,
                "top_speed_kmh": 25.0,
                "avg_speed_kmh": 6.0,
                "samples": 20,
                "sprints": 7,
                "high_speed_m": 60.0,
                "zones": {"run": 30.0, "walk": 5.0},
            },
        ]
        self.analysis = {
            "oyuncular": players,
            "takimlar": {},
            "isi_haritasi": {"oyuncular": {"1": {"5": 3}, "2": {"7": 2}}},
            "en_cok_kosan": players[:],
            "en_hizli": players[:],
        }
        self.analysis_path = db.ANALYSIS_DIR / str(self.group_id) / "remap.json"
        self.analysis_path.parent.mkdir(parents=True, exist_ok=True)
        self.analysis_path.write_text(json.dumps(self.analysis), encoding="utf-8")

        db.update_match(
            self.match_id,
            status="finished",
            analysis_path=str(self.analysis_path),
            stats={"players": players, "teams": {}},
        )
        db.set_match_names(self.match_id, self.group_id, {1: "Ali", 2: "Veli"})
        db.add_event(self.match_id, 1, 0, "goal", 5.0)
        db.add_event(self.match_id, 2, 0, "assist", 6.0)

        self.card_dir = cards.CARD_ROOT / self.match_id
        self.card_dir.mkdir()
        for tracker_id in (1, 2):
            (self.card_dir / f"{tracker_id}-fi-v2.png").write_bytes(b"png")
        (server.REPORTS_DIR / f"{self.match_id}-fi.png").write_bytes(b"png")
        heat_dir = heatmap.CACHE_DIR / self.match_id
        heat_dir.mkdir()
        (heat_dir / "all.png").write_bytes(b"png")

        group = {"id": self.group_id, "name": "Regresyon", "username": "regresyon"}
        patcher = mock.patch.object(server, "current_group", lambda request: group)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_swap_returns_200_and_moves_names_events_and_caches(self) -> None:
        result = server.remap_player(
            self.match_id, server.RemapPayload(source=1, target=2), None
        )

        rows = {row["tracker_id"]: row for row in result["players"]}
        self.assertEqual(set(rows), {1, 2})
        self.assertEqual(rows[1]["distance_m"], 200.0)
        self.assertEqual(rows[2]["distance_m"], 100.0)

        self.assertEqual(db.match_names(self.match_id), {1: "Veli", 2: "Ali"})
        events = {event["kind"]: event for event in db.list_events(self.match_id)}
        self.assertEqual(events["goal"]["tracker_id"], 2)
        self.assertEqual(events["assist"]["tracker_id"], 1)

        stored = json.loads(self.analysis_path.read_text(encoding="utf-8"))
        stored_rows = {row["tracker_id"]: row["distance_m"] for row in stored["oyuncular"]}
        self.assertEqual(stored_rows, {1: 200.0, 2: 100.0})
        heat = stored["isi_haritasi"]["oyuncular"]
        self.assertEqual(heat["1"], {"7": 2})
        self.assertEqual(heat["2"], {"5": 3})
        self.assertEqual(stored["takimlar"]["0"]["distance_m"], 300.0)

        self.assertFalse((cards.CARD_ROOT / self.match_id).exists())
        self.assertEqual(list(server.REPORTS_DIR.glob(f"{self.match_id}-*.png")), [])
        self.assertFalse((heatmap.CACHE_DIR / self.match_id).exists())

        match = db.get_match(self.match_id, self.group_id)
        self.assertEqual(match["stats"]["players"][0]["tracker_id"], 1)
        self.assertEqual(match["stats"]["players"][0]["distance_m"], 200.0)

    def test_invalid_remap_requests_return_400(self) -> None:
        cases = {
            "aynı oyuncu": server.RemapPayload(source=1, target=1),
            "maçta olmayan oyuncu": server.RemapPayload(source=1, target=9),
            "geçersiz mod": server.RemapPayload(source=1, target=2, mode="shuffle"),
        }
        for label, payload in cases.items():
            with self.subTest(label), self.assertRaises(server.HTTPException) as raised:
                server.remap_player(self.match_id, payload, None)
            self.assertEqual(raised.exception.status_code, 400)

        # Reddedilen istekler veriyi değiştirmemeli.
        self.assertEqual(db.match_names(self.match_id), {1: "Ali", 2: "Veli"})
        self.assertEqual(len(db.list_events(self.match_id)), 2)

    def test_unknown_match_returns_404(self) -> None:
        with self.assertRaises(server.HTTPException) as raised:
            server.remap_player(
                "olmayan-mac", server.RemapPayload(source=1, target=2), None
            )

        self.assertEqual(raised.exception.status_code, 404)

    def test_live_match_remap_returns_409(self) -> None:
        db.update_match(self.match_id, status="live")
        with mock.patch.object(live, "get", lambda match_id: object()):
            with self.assertRaises(server.HTTPException) as raised:
                server.remap_player(
                    self.match_id,
                    server.RemapPayload(source=1, target=2, mode="merge"),
                    None,
                )

        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(db.match_names(self.match_id), {1: "Ali", 2: "Veli"})
        self.assertEqual(len(db.list_events(self.match_id)), 2)

    def test_merge_folds_source_into_target(self) -> None:
        result = server.remap_player(
            self.match_id,
            server.RemapPayload(source=1, target=2, mode="merge"),
            None,
        )

        self.assertEqual(len(result["players"]), 1)
        merged = result["players"][0]
        self.assertEqual(merged["tracker_id"], 2)
        self.assertEqual(merged["distance_m"], 300.0)
        self.assertEqual(merged["top_speed_kmh"], 25.0)
        self.assertEqual(merged["avg_speed_kmh"], 5.7)
        self.assertEqual(merged["samples"], 30)
        self.assertEqual(merged["sprints"], 10)
        self.assertEqual(merged["high_speed_m"], 100.0)
        self.assertEqual(
            merged["zones"], {"walk": 5.0, "jog": 0.0, "run": 80.0, "sprint": 10.0}
        )
        self.assertEqual(
            result["teams"]["0"], {"players": 1, "distance_m": 300.0, "top_speed_kmh": 25.0}
        )

        self.assertEqual(db.match_names(self.match_id), {2: "Veli"})
        self.assertEqual(
            [event["tracker_id"] for event in db.list_events(self.match_id)], [2, 2]
        )

        stored = json.loads(self.analysis_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["oyuncular"], result["players"])
        self.assertEqual(stored["isi_haritasi"]["oyuncular"], {"2": {"5": 3, "7": 2}})
        for key in ("en_cok_kosan", "en_hizli"):
            self.assertEqual([row["tracker_id"] for row in stored[key]], [2])

        self.assertFalse((cards.CARD_ROOT / self.match_id).exists())
        self.assertEqual(list(server.REPORTS_DIR.glob(f"{self.match_id}-*.png")), [])
        self.assertFalse((heatmap.CACHE_DIR / self.match_id).exists())

        match = db.get_match(self.match_id, self.group_id)
        self.assertEqual(match["stats"]["players"], result["players"])

    def test_merge_moves_source_name_when_target_is_unnamed(self) -> None:
        db.set_match_names(self.match_id, self.group_id, {1: "Ali"})
        server.remap_player(
            self.match_id,
            server.RemapPayload(source=1, target=2, mode="merge"),
            None,
        )

        self.assertEqual(db.match_names(self.match_id), {2: "Ali"})


class CardPrebuildRaceTest(TempDataTestCase):
    """Silinen maç için kart üretimi dosya/klasör bırakmamalı."""

    def setUp(self) -> None:
        super().setUp()
        self.match_id = self.create_match()
        self.saves: list[int] = []

        def fake_save(
            match_id: str,
            tracker_id: int,
            lang: str,
            player: dict,
            meta: dict,
            awards=None,
            heat_png=None,
            heat_box=None,
        ) -> None:
            path = cards.CARD_ROOT / match_id / f"{tracker_id}-{lang}-v2.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png")
            self.saves.append(tracker_id)
            if len(self.saves) == 2:
                # Üretim sürerken maç silinir: eski kod artıkları bırakıyordu.
                db.delete_match(match_id, self.group_id)
            time.sleep(0.02)

        for name, value in (
            ("match_awards", lambda players: {}),
            ("build_meta", lambda **kwargs: {}),
            ("save_card", fake_save),
        ):
            patcher = mock.patch.object(cards, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.payload = {
            "players": [{"tracker_id": i, "team": 0} for i in range(1, 9)],
            "teams": {"0": {"distance_m": 10.0}},
            "elapsed_s": 60,
        }

    def make_runner(self) -> live.LiveMatch:
        runner = live.LiveMatch.__new__(live.LiveMatch)
        runner.id = self.match_id
        runner.group_id = self.group_id
        runner.title = "Regresyon maçı"
        runner.started_at = time.time()
        return runner

    def test_cards_are_removed_when_match_deleted_during_generation(self) -> None:
        runner = self.make_runner()
        folder = cards.CARD_ROOT / self.match_id
        runner._prebuild_cards(self.payload)

        deadline = time.time() + 5
        while time.time() < deadline and (len(self.saves) < 8 or folder.exists()):
            time.sleep(0.02)

        self.assertGreaterEqual(len(self.saves), 3, "silme üretim sürerken olmalıydı")
        self.assertFalse(folder.exists(), "silinmiş maçın kartları/klasörü kaldı")
        self.assertIsNone(db.get_match(self.match_id, self.group_id))

    def test_no_cards_when_match_already_deleted(self) -> None:
        db.delete_match(self.match_id, self.group_id)
        runner = self.make_runner()
        runner._prebuild_cards(self.payload)
        time.sleep(0.3)

        self.assertEqual(self.saves, [])
        self.assertFalse((cards.CARD_ROOT / self.match_id).exists())


if __name__ == "__main__":
    unittest.main()
