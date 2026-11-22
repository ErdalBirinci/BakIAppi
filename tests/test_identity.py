"""Oyuncu kimliği (görünüm profili) birim testleri.

Gerçek SigLIP gömlemleri yerine, takım ortalaması çıkarıldığında ayrışan
küçük vektörler kullanılır. Amaç, kimlik onarımının kurallarını kilitlemek:
  * kopan ID aynı oyuncuya bağlanır,
  * o anda sahada görünen oyuncular birbirine karışmaz,
  * farklı takımdaki ve benzemeyen oyuncular birleştirilmez,
  * bağlantı sonraki karelerde kalıcıdır.
"""

from __future__ import annotations

import unittest

import numpy as np

from app.identity import PlayerIdentity

PLAYER_A = np.array([1.0, 0.0, 0.0, 0.05], dtype=np.float32)
PLAYER_B = np.array([0.0, 1.0, 0.0, 0.05], dtype=np.float32)
PLAYER_A_AGAIN = np.array([0.97, 0.03, 0.0, 0.06], dtype=np.float32)
PLAYER_B_AGAIN = np.array([0.03, 0.97, 0.0, 0.06], dtype=np.float32)
PLAYER_C = np.array([0.05, 0.0, 1.0, 0.0], dtype=np.float32)
PLAYER_D = np.array([0.0, 0.05, 0.0, 1.0], dtype=np.float32)
PLAYER_C_AGAIN = np.array([0.06, 0.0, 0.97, 0.0], dtype=np.float32)
STRANGER = np.array([0.0, 0.0, 0.7, 0.7], dtype=np.float32)


class PlayerIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = PlayerIdentity(threshold=0.55, enabled=True)
        self.resolved = self.resolve(
            [11, 12, 21, 22],
            [PLAYER_A, PLAYER_B, PLAYER_C, PLAYER_D],
            [0, 0, 1, 1],
        )

    def resolve(self, tracker_ids, embeddings, teams) -> list[int]:
        return [
            int(value)
            for value in self.identity.resolve(tracker_ids, embeddings, teams)
        ]

    def test_first_frame_keeps_tracker_ids(self) -> None:
        self.assertEqual(self.resolved, [11, 12, 21, 22])
        self.assertEqual(self.identity.merge_count, 0)

    def test_returning_player_reuses_original_id(self) -> None:
        resolved = self.resolve([12, 99], [None, PLAYER_A_AGAIN], [0, 0])

        self.assertEqual(resolved, [12, 11])
        self.assertEqual(self.identity.alias, {99: 11})
        self.assertEqual(self.identity.merge_count, 1)
        self.assertEqual(self.identity.merges[0]["from"], 99)
        self.assertEqual(self.identity.merges[0]["to"], 11)
        self.assertGreaterEqual(self.identity.merges[0]["score"], 0.55)

    def test_alias_persists_on_later_frames(self) -> None:
        self.resolve([12, 99], [None, PLAYER_A_AGAIN], [0, 0])

        self.assertEqual(self.resolve([99], [None], [0]), [11])
        self.assertEqual(self.identity.merge_count, 1)

    def test_visible_teammate_blocks_merge(self) -> None:
        # A ve B aynı karede görünürken B'ye benzeyen yeni bir ID, B'ye bağlanmamalı.
        resolved = self.resolve(
            [11, 12, 98], [None, None, PLAYER_B_AGAIN], [0, 0, 0]
        )

        self.assertEqual(resolved, [11, 12, 98])
        self.assertEqual(self.identity.merge_count, 0)

    def test_other_team_is_never_merged(self) -> None:
        resolved = self.resolve([97], [PLAYER_C_AGAIN], [0])

        self.assertEqual(resolved, [97])
        self.assertEqual(self.identity.merge_count, 0)

    def test_unfamiliar_appearance_stays_separate(self) -> None:
        resolved = self.resolve([96], [STRANGER], [0])

        self.assertEqual(resolved, [96])
        self.assertEqual(self.identity.merge_count, 0)

    def test_missing_embedding_does_not_merge(self) -> None:
        resolved = self.resolve([95], [None], [0])

        self.assertEqual(resolved, [95])
        self.assertEqual(self.identity.merge_count, 0)

    def test_disabled_identity_keeps_raw_ids(self) -> None:
        identity = PlayerIdentity(threshold=0.55, enabled=False)
        identity.enroll(11, PLAYER_A, 0)

        resolved = identity.resolve([99], [PLAYER_A_AGAIN], [0])

        self.assertEqual(int(resolved[0]), 99)
        self.assertEqual(identity.merge_count, 0)


if __name__ == "__main__":
    unittest.main()
