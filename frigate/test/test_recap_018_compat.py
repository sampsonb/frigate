"""Recap still accepts the live 0.17 config and the files already on disk.

The production config is ``version: 0.17-0`` with the recap block captured
on 2026-11-09 (global interval and retention, front at 01:37, aqara at
01:52 with its own labels and parked cars off). 0.18 rewrites that file
to ``0.18-0`` on startup. Recap adds no table of its own and keeps reading
the same manifest, schedule, interval, and cutout-cache files.
"""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from frigate.config import FrigateConfig
from frigate.recap import cutcache
from frigate.recap.manager import RecapMaintainer, RecapManager
from frigate.recap.storage import list_visible_recaps, purge_expired, safe_id
from frigate.util.config import migrate_frigate_config

# Live sections from the 2026-11-09 install, plus the prepared snippet's
# timezone. Detector paths match the OpenVINO GPU model shipped in the image.
# The label file is a stand-in so parsing does not need the image contents.
_CONFIG = """
version: 0.17-0

mqtt:
  host: mqtt

ui:
  timezone: America/Chicago

recap:
  enabled: false
  window_hours: 24
  max_window_hours: 48
  interval_minutes: 30
  retain_days: 14

detectors:
  ov:
    type: openvino
    device: GPU

model:
  path: /openvino-model/ssdlite_mobilenet_v2.xml
  labelmap_path: {labelmap}
  width: 300
  height: 300
  input_tensor: nhwc
  input_pixel_format: bgr

cameras:
  front:
    ffmpeg:
      inputs:
        - path: rtsp://10.0.0.1:554/video
          roles:
            - detect
    detect:
      width: 1920
      height: 1080
      fps: 5
    recap:
      enabled: true
      schedule: "01:37"
  aqara:
    ffmpeg:
      inputs:
        - path: rtsp://10.0.0.2:554/video
          roles:
            - detect
    detect:
      width: 1920
      height: 1080
      fps: 5
    recap:
      enabled: true
      schedule: "01:52"
      parked_cars: false
      labels:
        - person
        - dog
        - cat
        - package
        - amazon
        - ups
        - fedex
        - usps
        - dhl
"""

_AQARA_LABELS = [
    "person",
    "dog",
    "cat",
    "package",
    "amazon",
    "ups",
    "fedex",
    "usps",
    "dhl",
]


def _write_manifest(root: Path, camera: str, recap_id: str, **fields: object) -> None:
    folder = root / camera / recap_id
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": recap_id,
        "camera": camera,
        "status": "complete",
        "message": "Ready",
        "created": 1,
        "after": 0.0,
        "before": 1.0,
        "reason": "manual",
        "event_count": 1,
    }
    payload.update(fields)
    (folder / "manifest.json").write_text(json.dumps(payload))
    (folder / "video.mp4").write_bytes(b"")
    (folder / "thumb.jpg").write_bytes(b"")


class TestRecapConfigMigration(unittest.TestCase):
    def test_live_017_recap_config_validates_after_018_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labelmap = root / "labels.txt"
            labelmap.write_text("person\ncar\ndog\n")
            config_path = root / "config.yml"
            config_path.write_text(_CONFIG.format(labelmap=labelmap))

            with patch("frigate.util.config.CONFIG_DIR", str(root)):
                migrate_frigate_config(str(config_path))

            migrated = config_path.read_text()
            self.assertIn("version: 0.18-0", migrated)
            self.assertIn("01:37", migrated)
            self.assertIn("01:52", migrated)
            self.assertIn("parked_cars", migrated)
            self.assertIn("interval_minutes: 30", migrated)
            self.assertIn("retain_days: 14", migrated)
            # The backup copy of the pre-migration file stays put.
            self.assertTrue((root / "backup_config.yaml").is_file())

            config = FrigateConfig.parse_yaml(migrated)
            self.assertEqual(
                config.model.path, "/openvino-model/ssdlite_mobilenet_v2.xml"
            )
            self.assertEqual(config.detectors["ov"].type, "openvino")
            self.assertEqual(config.detectors["ov"].device, "GPU")
            self.assertFalse(config.recap.enabled)
            self.assertEqual(config.recap.window_hours, 24)
            self.assertEqual(config.recap.max_window_hours, 48)
            self.assertEqual(config.recap.interval_minutes, 30)
            self.assertEqual(config.recap.retain_days, 14)
            self.assertEqual(config.ui.timezone, "America/Chicago")

            front = config.cameras["front"].recap
            aqara = config.cameras["aqara"].recap
            self.assertTrue(front.enabled)
            self.assertEqual(front.schedule, "01:37")
            self.assertTrue(front.parked_cars)
            self.assertEqual(front.interval_minutes, 30)
            self.assertEqual(front.retain_days, 14)
            self.assertTrue(aqara.enabled)
            self.assertEqual(aqara.schedule, "01:52")
            self.assertFalse(aqara.parked_cars)
            self.assertEqual(aqara.labels, _AQARA_LABELS)
            self.assertEqual(aqara.interval_minutes, 30)

            # Running migration again must leave the recap block alone.
            with patch("frigate.util.config.CONFIG_DIR", str(root)):
                migrate_frigate_config(str(config_path))
            again = FrigateConfig.parse_yaml(config_path.read_text())
            self.assertEqual(again.cameras["aqara"].recap.labels, _AQARA_LABELS)
            self.assertEqual(again.cameras["front"].recap.schedule, "01:37")

    def test_recap_adds_no_database_migration(self) -> None:
        root = Path(__file__).resolve().parents[2] / "migrations"
        names = sorted(path.name for path in root.glob("*.py"))
        self.assertIn("033_create_export_case_table.py", names)
        self.assertIn("034_add_export_case_to_exports.py", names)
        self.assertIn("035_add_motion_heatmap.py", names)
        self.assertNotIn("036_create_recap_table.py", names)
        for path in root.glob("*.py"):
            text = path.read_text(errors="replace").lower()
            self.assertNotIn("recap", text, path.name)


class TestExistingRecapFiles(unittest.TestCase):
    def test_production_layout_lists_newest_window_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".staging").mkdir()
            (root / ".schedule.json").write_text(
                json.dumps(
                    {
                        "front:2026-11-09:01:37": 1762670220,
                        "aqara:2026-11-09:01:52": 1762671120,
                    }
                )
            )
            (root / ".interval.json").write_text(
                json.dumps(
                    {
                        "front:2026-11-09-09:00": 1762696800,
                        "aqara:2026-11-09-09:00": 1762696800,
                    }
                )
            )
            _write_manifest(
                root,
                "front",
                "front_day",
                kind="day:2026-11-09",
                reason="schedule",
                before=300.0,
                created=30,
            )
            _write_manifest(
                root,
                "front",
                "front_rolling",
                kind="rolling",
                reason="rolling",
                before=500.0,
                created=50,
            )
            _write_manifest(
                root,
                "front",
                "front_rolling_r6h",
                kind="rolling-archive:front_rolling_r6h",
                reason="rolling-archive",
                before=400.0,
                created=40,
            )
            _write_manifest(
                root,
                "aqara",
                "aqara_30m",
                kind="rolling-30m",
                reason="interval",
                before=450.0,
                created=45,
            )
            # Dot directories are not cameras. A cutcache entry is not a recap.
            cache_key = "ab" + ("c" * 40)
            payload = {
                "v": cutcache.CACHE_VERSION,
                "status": "ok",
                "frames": [],
                "boxes": [(0.1, 0.2, 0.3, 0.4)],
                "times": [12.5],
                "context_first": None,
                "context_last": None,
                "still": False,
                "shift": -0.25,
            }
            cache_path = root / ".cutcache" / cache_key[:2] / f"{cache_key}.pkl"
            cache_path.parent.mkdir(parents=True)
            cache_path.write_bytes(zlib.compress(pickle.dumps(payload, protocol=4), 3))

            with patch("frigate.recap.storage.RECAP_DIR", str(root)):
                listed = list_visible_recaps([])

            self.assertEqual(
                [item["id"] for item in listed],
                ["front_rolling", "aqara_30m", "front_rolling_r6h", "front_day"],
            )
            self.assertEqual(
                [item["before"] for item in listed],
                sorted((item["before"] for item in listed), reverse=True),
            )
            loaded = cutcache.load(cache_key, root / ".cutcache")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.status, "ok")
            self.assertEqual(loaded.shift, -0.25)
            self.assertEqual(loaded.boxes, [(0.1, 0.2, 0.3, 0.4)])

            schedule = json.loads((root / ".schedule.json").read_text())
            interval = json.loads((root / ".interval.json").read_text())
            self.assertTrue(schedule["front:2026-11-09:01:37"])
            self.assertIn("aqara:2026-11-09-09:00", interval)

    def test_retention_cannot_walk_out_of_the_recap_directory(self) -> None:
        self.assertFalse(safe_id(".."))
        self.assertFalse(safe_id("."))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = (
                root / "recordings" / "2026-11-09" / "12" / "front" / "00.00.mp4"
            )
            recording.parent.mkdir(parents=True)
            recording.write_bytes(b"recording-bytes")
            # Stored under a normal folder. The manifest names a parent path.
            _write_manifest(
                root / "recap",
                "front",
                "evil",
                id="recordings",
                created=1,
            )
            manifest = json.loads(
                (root / "recap" / "front" / "evil" / "manifest.json").read_text()
            )
            manifest["camera"] = ".."
            manifest["id"] = "recordings"
            (root / "recap" / "front" / "evil" / "manifest.json").write_text(
                json.dumps(manifest)
            )
            with patch("frigate.recap.storage.RECAP_DIR", str(root / "recap")):
                removed = purge_expired({"..": 14, "front": 14}, now=10**12)
            self.assertEqual(removed, 0)
            self.assertTrue(recording.is_file())
            self.assertEqual(recording.read_bytes(), b"recording-bytes")


class TestRecapFollowsConfigHolder(unittest.TestCase):
    def test_manager_and_maintainer_read_the_swapped_config(self) -> None:
        class Holder:
            def __init__(self, config: object) -> None:
                self.config = config

        first: object = {"boot": True}
        second: object = {"saved": True}
        manager = RecapManager(first)  # type: ignore[arg-type]
        maintainer = RecapMaintainer(first, stop_event=None, manager=manager)  # type: ignore[arg-type]
        self.assertIs(manager.config, first)
        self.assertIs(maintainer.config, first)
        manager.config_holder = Holder(second)
        maintainer.config_holder = Holder(second)
        self.assertIs(manager.config, second)
        self.assertIs(maintainer.config, second)


if __name__ == "__main__":
    unittest.main()
