import json
from pathlib import Path
import tempfile
import unittest

from mypic_bot.dashboard import (
    _image_content_type,
    read_decisions,
    resolve_image_for_selection,
)


class DashboardDataTests(unittest.TestCase):
    def test_detects_image_type_without_file_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "cached-image"
            image.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 12)
            self.assertEqual(_image_content_type(image), "image/jpeg")

    def test_reads_newest_first_and_hides_filesystem_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            image = images / "chosen.webp"
            image.write_bytes(b"image")
            log = root / "decisions.jsonl"
            log.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {
                            "selection_id": "older",
                            "result": {"status": "stayed_silent"},
                        },
                        {
                            "selection_id": "newer",
                            "candidates": [{"local_path": str(image)}],
                            "result": {
                                "status": "selected",
                                "path": str(image),
                            },
                        },
                    )
                ),
                encoding="utf-8",
            )
            records = read_decisions(log, 10, images)
        self.assertEqual(
            [record["selection_id"] for record in records],
            ["newer", "older"],
        )
        self.assertNotIn("path", records[0]["result"])
        self.assertNotIn("local_path", records[0]["candidates"][0])
        self.assertEqual(
            records[0]["result"]["image_url"],
            "/api/image/newer",
        )

    def test_only_serves_images_inside_configured_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            allowed = images / "allowed.webp"
            allowed.write_bytes(b"allowed")
            outside = root / "outside.webp"
            outside.write_bytes(b"outside")
            log = root / "decisions.jsonl"
            log.write_text(
                "\n".join(
                    (
                        json.dumps(
                            {
                                "selection_id": "allowed",
                                "result": {"path": str(allowed)},
                            }
                        ),
                        json.dumps(
                            {
                                "selection_id": "outside",
                                "result": {"path": str(outside)},
                            }
                        ),
                    )
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                resolve_image_for_selection(log, "allowed", images),
                allowed,
            )
            self.assertIsNone(
                resolve_image_for_selection(log, "outside", images)
            )


if __name__ == "__main__":
    unittest.main()
