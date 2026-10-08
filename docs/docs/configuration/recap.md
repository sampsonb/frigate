---
id: recap
title: Recap
---

Recap builds a short video synopsis of a long stretch of footage. Every tracked object from the window is cut out and played over one background, with a stationary time label. Hours of a camera can be reviewed in about two minutes. The idea is the same as FLIR RapidRecap.

Recap is off unless you enable it. Leaving it out of the config does not change detection, recording, or review.

## Configuration

```yaml
recap:
  enabled: true
  # Daily local time. Omit this to generate recaps only from the UI.
  schedule: "02:00"
  # Hours of footage each scheduled recap covers.
  window_hours: 24
  retain_days: 14
```

The same block can be set on one camera. A camera value replaces the global value for that field. `enabled: false` is the default, so a stock config never starts a job.

Useful knobs:

| Field | Default | Purpose |
| --- | --- | --- |
| `schedule` | none | Daily `HH:MM` in `ui.timezone`. The container clock is often UTC, and this time is not UTC unless that setting is unset. Frigate has to be running within about 90 seconds of this time. |
| `window_hours` | 24 | How far back a scheduled recap looks. |
| `max_window_hours` | 48 | Longest on-demand range. |
| `labels` | people, vehicles, animals, delivery names | Tracked labels to consider. |
| `target_length` | 120 | Aim for this many seconds. Busy hours run longer rather than merging objects into one label. |
| `label_opacity` | 0.5 | Time-label background. |
| `max_labels` | 8 | Most labels on screen at once. |
| `fade_seconds` | 0.35 | Fade for the ghost, the leader line, and the label. |
| `retain_days` | 14 | Delete local recaps after this many days. `0` keeps them. |
| `min_show_seconds` | 2.5 | Shortest time a ghost stays readable. |
| `output_fps` | 12 | Synopsis frame rate. |
| `max_width` | 1280 | Frames are scaled down to this. |
| `pause_seconds` | 0.02 | Sleep between events so live detection keeps the CPU. |
| `archive` | off | Optional older media. See below. |

Generation runs one job at a time, at a lower CPU priority. Another request waits in a queue. Cancel from the Recap page.

Files are stored at `/media/frigate/recap/<camera>/<id>/` (`manifest.json`, `video.mp4`, `thumb.jpg`).

## What gets drawn

- People are blue, vehicles orange, deliveries magenta, animals green, and a parked car someone got into or out of is teal.
- The legend checkboxes hide labels and list rows. The ghosts are already part of the video.
- A vehicle that barely moves is left out, unless a person gets in or out. That car is drawn once, labeled `12:26 PM got out` or `got in`, with a dotted line to the person.
- A delivery is a Frigate+ attribute or label (`fedex`, `ups`, `amazon`, `package`, and the other carrier names) when one is present. Otherwise semantic search is used if it is enabled.
- A person overlapping a dog or cat is drawn as an animal, which is how a dog walker stays one green label.
- Anything that moves with a person, and is connected to them against a clean background, is part of their cutout. That includes a golf cart, bicycle, scooter, or stroller the detector does not know as its own object.
- Times are local, without seconds (`5:50 PM`). If two labels fall in the same clock minute, those labels show seconds (`5:50:12 PM`, `5:50:47 PM`). Labels are not grouped with a count.

Click a time label to jump the synopsis. Double-click it to open that event's clip, with download and previous/next through the events that match the legend. If Frigate has already deleted the recording, and `recap.archive` is set, the player uses the archived file for the same event.

## Archive

Frigate's own retention is often 10 to 14 days. A nightly job outside Frigate can keep recordings for about two months and events for a year or more. Point recap at that copy and it will play clips, and list recaps, after the live files are gone.

```yaml
recap:
  enabled: true
  archive:
    # Mount of the external drive. Same layout as /media/frigate.
    path: /mnt/frigate-archive
    # Optional. Used when the drive is not mounted and the tree is served over HTTP.
    url: https://archive.example/frigate
```

Both fields are optional. With neither set, recap never reads the archive.

The archive root should keep Frigate's paths:

- `recordings/YYYY-MM-DD/HH/<camera>/MM.SS.mp4` in UTC, the same names Frigate writes
- `clips/<camera>-<event id>.jpg` snapshots
- `recap/<camera>/<id>/manifest.json`, `video.mp4`, and `thumb.jpg` if the nightly copy includes recaps

Also include a per-day index and, if you have them, dated database copies:

- `index/YYYY-MM-DD.json` (`.jsonl` and `.csv` are read too, as are `YYYY-MM-DD/index.json`)
- `db/frigate-YYYY-MM-DD.db`

An index row needs the event id, camera, label, start, end, and archive-relative paths. A clip path or the recording segments both work. Paths may also be the original `/media/frigate/...` location. They are rewritten onto the archive root. Paths that leave the archive (`..`) are ignored.

```json
[
  {
    "id": "1728330612.481-a1b2c3",
    "camera": "front",
    "label": "person",
    "start": 1728330612.481,
    "end": 1728330640.2,
    "paths": [
      "recordings/2024-10-07/20/front/50.12.mp4",
      "recordings/2024-10-07/20/front/50.22.mp4"
    ],
    "snapshot": "clips/front-1728330612.481-a1b2c3.jpg"
  }
]
```

`event_id`, `start_time`, `end_time`, `files`, and `clip` are accepted names for the same fields. CSV uses a header row, with several paths separated by `|`.

The nightly index at `index/YYYY-MM-DD.json` is a list of rows. Each row has `id`, `camera`, `label`, `start`, `end` (epoch seconds), `paths` (for example `events/<review id>.mp4`), and `snapshot`. Frigate-built recaps copied to `recap/` use Frigate's own `<camera>/<id>/manifest.json` layout.

When the index does not list files, recap looks up the event in a dated database and then the UTC recording tree. One file is served directly. Several files are joined the same way Frigate builds an event clip. Every file is checked again immediately before it is opened. If one was deleted after the index was read, the player gets "This clip is no longer archived" instead of a server error. If only the snapshot is left, the player shows the snapshot.

Archived recaps show in the Recap list with an Archive badge. They are not deleted from the UI. The local `retain_days` cleanup does not touch the archive.

If you only set `url`, clip fallback reads `index/YYYY-MM-DD.json` from that host, and the list reads `recap/index.json`. Playing a recording that spans multiple files needs `path`, because Frigate has to join them.

## Install on an existing 0.17.2 container

Build the overlay on the host. The repository is public, so this does not need a GitHub login. It starts from stock `ghcr.io/blakeblackshear/frigate:0.17.2` (pinned by digest) and does not compile Frigate. On a Beelink that already has that image, expect about 15 to 25 minutes. Most of that is `npm install` and the web UI production build. The first run also downloads the `node:20` image used only for that build.

```bash
docker build -t frigate:recap-0.17.2 -f docker/recap/Dockerfile https://github.com/sampsonb/frigate.git#cursor/recap-3a01
```

In `docker-compose.yml`, point the service at the local tag. Leave the config volume, media volume, and `/dev/dri` device (OpenVINO on `/dev/dri/renderD128`) as they are. Do not `docker compose pull` this tag. It is not in a registry.

```yaml
image: frigate:recap-0.17.2
```

```bash
docker compose up -d
```

Add `recap:` to `config.yml`, including `ui.timezone`, and restart once more. Until `recap.enabled` is true, behavior matches stock 0.17.2.

Roll back by restoring the stock image and recreating the container:

```yaml
image: ghcr.io/blakeblackshear/frigate:0.17.2
```

```bash
docker compose up -d
```

Recap files under `/media/frigate/recap` are left in place and are ignored by the stock image.

A GitHub Actions workflow on this fork can also push `ghcr.io/sampsonb/frigate:recap-0.17.2`. A new GHCR package is private by default, so `docker pull` of that tag fails until the package is made public. The `docker build` command above does not depend on that.

## Performance

Recap is meant for a small Intel machine that is also running detection. It does not load a second detector. Cutouts are motion against a background plate, on the CPU, with a short pause between events. Output is 1280px wide H.264. One recap runs at a time, and the worker lowers its priority.

Semantic search, when enabled, only looks at the nearest thumbnails Frigate already computed. It is skipped when semantic search is off.

## Limitations

- Filters hide labels and the event list. They do not erase ghosts already drawn into the video.
- A parked car is shown only when Frigate tracked that stationary vehicle. A car the detector never saw cannot be labeled.
- Cutouts are weaker than a segmentation model on a cluttered background. A connected moving object is included. A shadow or a full-frame lighting change is not.
- The schedule is a single daily time, not a cron expression. If Frigate is down at that minute, that day's recap is skipped.
- Label times and the nightly schedule use `ui.timezone`. The container clock is often UTC. If `ui.timezone` is unset, both fall back to the container's local time.
- At most `max_events` events are processed (default 400), oldest first after parked vehicles are set aside.
- An archive URL alone cannot join multiple recording files. Mount `archive.path` for that.
