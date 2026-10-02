import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from sae_dashboard.manifests import DATA_ROOT, load_manifest, load_manifests


class ManifestTests(unittest.TestCase):
    def test_repository_code_manifest_has_twenty_real_pairs(self):
        manifest = load_manifest(DATA_ROOT / "cwe_c_pairs_dataset/manifest.json")
        self.assertEqual(len(manifest.pairs), 20)
        self.assertIn("#include", manifest.pairs[0].a.text)
        self.assertIsNone(manifest.pairs[0].a.image)
        self.assertNotEqual(manifest.pairs[0].a.text, manifest.pairs[0].b.text)

    def test_modalities_and_image_integrity(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("sae_dashboard.manifests.DATA_ROOT", Path(temp).resolve()),
        ):
            root = Path(temp).resolve()
            image = root / "image.png"
            Image.new("RGB", (2, 2), "blue").save(image)
            raw = {
                "version": 1,
                "name": "test",
                "pairs": [
                    {
                        "A": {"text": "", "image": "image.png"},
                        "B": {"text": "test", "image": ""},
                    }
                ],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(raw))
            manifest = load_manifest(path)
            self.assertEqual(manifest.pairs[0].a.load_image().size, (2, 2))
            Image.new("RGB", (2, 2), "red").save(image)
            with self.assertRaisesRegex(ValueError, "changed"):
                manifest.pairs[0].a.load_image()
            for value in ({}, {"text": " ", "image": ""}):
                raw["pairs"][0]["A"] = value
                path.write_text(json.dumps(raw))
                with self.assertRaisesRegex(ValueError, "needs text"):
                    load_manifest(path)

    def test_relative_paths_work_for_uploaded_json_and_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            data = root / "data"
            (data / "code").mkdir(parents=True)
            (data / "code/a.txt").write_text("A")
            (root / "outside.txt").write_text("secret")
            raw = {
                "version": 1,
                "name": "test",
                "asset_root": "code",
                "pairs": [{"A": {"text_file": "a.txt"}, "B": {"text": "B"}}],
            }
            upload = root / "upload.json"
            with patch("sae_dashboard.manifests.DATA_ROOT", data):
                upload.write_text(json.dumps(raw))
                self.assertEqual(load_manifest(upload).pairs[0].a.text, "A")
                for value in ("../../outside.txt", str(root / "outside.txt")):
                    raw["pairs"][0]["A"]["text_file"] = value
                    upload.write_text(json.dumps(raw))
                    with self.assertRaisesRegex(ValueError, "inside"):
                        load_manifest(upload)
                (data / "code/link.txt").symlink_to(root / "outside.txt")
                raw["pairs"][0]["A"]["text_file"] = "link.txt"
                upload.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    load_manifest(upload)

    def test_duplicate_manifest_names_rejected(self):
        path = str(DATA_ROOT / "cwe_c_pairs_dataset/manifest.json")
        with self.assertRaisesRegex(ValueError, "unique name"):
            load_manifests([path, path])
