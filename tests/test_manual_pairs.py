import copy
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import unittest
from PIL import Image
from sae_dashboard.manifests import DATA_ROOT, MAX_PAIRS, Condition, Manifest, Pair
from sae_dashboard.manual_pairs import change_pair, read_pair, manual_condition
from sae_dashboard.session import Session
from sae_dashboard import workflow


class ManualPairTests(unittest.TestCase):
    def test_text_image_and_mixed_pairs_work_without_disk_writes(self):
        state = Session()
        picture = Image.new("RGB", (3, 2), "blue")
        with patch(
            "pathlib.Path.open",
            side_effect=AssertionError("Manual pairs must remain in memory"),
        ):
            change_pair(state, "add", a_text="A", b_text="B")
            change_pair(state, "add", a_image=picture, b_image=picture)
            change_pair(
                state, "add", a_text="Look at this", a_image=picture, b_text="Text only"
            )
            text, loaded, _, right = read_pair(state, "Pair 3")
            self.assertEqual(text, "Look at this")
            self.assertEqual(loaded.size, (3, 2))
            self.assertIsNone(right)
        self.assertEqual(len(state.manifests[0].pairs), 3)
        picture.putpixel((0, 0), (255, 0, 0))
        self.assertEqual(
            state.manual_pairs[1].a.load_image().getpixel((0, 0)), (0, 0, 255)
        )
        loaded.putpixel((0, 0), (0, 255, 0))
        self.assertEqual(
            state.manual_pairs[2].a.load_image().getpixel((0, 0)), (0, 0, 255)
        )
        serialized = json.dumps(state.manifests[0].metadata())
        self.assertIn("manual upload (session memory)", serialized)
        self.assertNotIn("image_bytes", serialized)

    def test_empty_conditions_fail_atomically(self):
        state = Session()
        state.profile_id = "unchanged"
        for values in ({"a_text": "  ", "b_text": "B"}, {"a_text": "A"}, {}):
            with self.assertRaisesRegex(ValueError, "needs text"):
                change_pair(state, "add", **values)
        self.assertEqual(state.profile_id, "unchanged")
        self.assertFalse(state.manifests)
        self.assertEqual(state.manual_next_id, 1)
        with self.assertRaisesRegex(ValueError, "image field"):
            manual_condition("A", "/untrusted/path")

    def test_edit_delete_invalidate_profile_but_preserve_other_inputs(self):
        state = Session()
        workflow.set_manifests(state, [str(DATA_ROOT / "scripts/cwe_120/manifest.json")])
        original = state.manifests[0]
        change_pair(state, "add", a_text="A", b_text="B")
        first_fingerprint = state.manifests[-1].fingerprint
        state.profile_id, state.vector_id = "old-profile", "old-vector"
        state.profile, state.vectors = {9: "old"}, {9: "old"}
        change_pair(state, "update", "Pair 1", a_text="New A", b_text="New B")
        self.assertNotEqual(first_fingerprint, state.manifests[-1].fingerprint)
        self.assertIsNone(state.profile_id)
        self.assertIsNone(state.vector_id)
        self.assertFalse(state.vectors)
        self.assertIs(state.manifests[0], original)
        change_pair(state, "remove", "Pair 1")
        self.assertEqual(state.manifests, (original,))
        self.assertFalse(state.manual_pairs)
        with self.assertRaisesRegex(ValueError, "existing manual pair"):
            change_pair(state, "remove", "Pair 1")
        change_pair(state, "add", a_text="A", b_text="B")
        self.assertEqual(state.manual_pairs[0].id, "Pair 2")

    def test_reload_manifests_preserves_manual_pairs(self):
        state = Session()
        change_pair(state, "add", a_text="A", b_text="B")
        for _ in range(2):
            rows = workflow.set_manifests(
                state, [str(DATA_ROOT / "scripts/cwe_120/manifest.json")]
            )
            self.assertEqual(sum(row[1] for row in rows), 21)
            self.assertEqual(len(state.manifests), 2)
            self.assertEqual(state.manifests[-1].pairs, state.manual_pairs)

    def test_combined_limit_and_name_collisions_leave_existing_state_intact(self):
        state = Session()
        state.manifests = (
            Manifest(
                "Full",
                tuple(
                    Pair(str(i), Condition("A"), Condition("B"))
                    for i in range(MAX_PAIRS)
                ),
                "hash",
            ),
        )
        original = state.manifests
        with self.assertRaisesRegex(ValueError, "at most"):
            change_pair(state, "add", a_text="A", b_text="B")
        self.assertIs(state.manifests, original)
        self.assertFalse(state.manual_pairs)
        state.manifests = (
            Manifest(
                "Manual pairs", (Pair("x", Condition("A"), Condition("B")),), "hash"
            ),
        )
        with self.assertRaisesRegex(ValueError, "reserved"):
            change_pair(state, "add", a_text="A", b_text="B")

    def test_sessions_are_isolated_and_concurrent_adds_have_unique_ids(self):
        first = Session()
        change_pair(first, "add", a_text="A", b_text="B")
        second = copy.deepcopy(first)
        with ThreadPoolExecutor(3) as pool:
            list(
                pool.map(
                    lambda n: change_pair(first, "add", a_text=str(n), b_text="B"),
                    range(3),
                )
            )
        self.assertEqual(len(first.manual_pairs), 4)
        self.assertEqual(len({p.id for p in first.manual_pairs}), 4)
        self.assertEqual(len(second.manual_pairs), 1)
        change_pair(second, "update", "Pair 1", a_text="Different", b_text="B")
        self.assertEqual(first.manual_pairs[0].a.text, "A")
