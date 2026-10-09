---
id: recap
title: Recap
---

Recap builds a short video synopsis of a long stretch of footage. Every tracked object from the window is cut out and played over a background from the same time of day, with a stationary time label. Hours of a camera can be reviewed in about two minutes. The idea is the same as FLIR RapidRecap.

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
| `interval_minutes` | off | Build a recap of the last this many minutes on that interval. The kind is `rolling-<minutes>m`, so the previous one of that kind is replaced when the new one is Ready. A run is skipped while that one is still queued or rendering, and a missed interval is not queued behind it. Separate from the rolling recap that stays ready. |
| `window_hours` | 24 | How far back a scheduled recap looks. |
| `max_window_hours` | 48 | Longest on-demand range. |
| `labels` | people, vehicles, animals, delivery names | Tracked labels to consider. |
| `target_length` | 120 | Aim for this many seconds. Busy hours run longer rather than merging objects into one label. |
| `label_opacity` | 0.5 | Time-label background. |
| `max_labels` | 8 | Most labels on screen at once. |
| `fade_seconds` | 0.2 | Fade for the ghost, the leader line, and the label. |
| `retain_days` | 14 | Delete local recaps after this many days. `0` keeps them. |
| `replace_superseded` | true | When a recap is Ready, delete older completed recaps of the same camera and kind. Last-N-hours buttons group by that length. An explicit range that is one local midnight-to-midnight day groups with the nightly recap for that date (`day:<date>`, including older `backfill-day:<date>` copies). Any other explicit range is kept on its own and is not treated as last-N hours. Running, failed, and archive copies are kept. |
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
- The background matches the lighting of the objects on screen. A frame with almost no color is treated as infrared. Color and infrared are never averaged into one plate. Within one lighting period the plate is the sample nearest those objects, refreshed about every 30 minutes. Objects stay in time order, so a dusk recap moves from afternoon to night, and the plate crossfades for about half a second when the lighting changes.
- A cutout that is still color on an infrared plate is turned gray, and its brightness and contrast are matched to the local plate. On an infrared or dark plate the cutout also gets a mild contrast boost and a thin outline in its category color, so a dark car on a dark road stays visible. Only an infrared vehicle is tightened against the plate, and only inside the padded cutout window. Lit ground is left out. A mask that is mostly the plate is left out. Day cutouts and people are not rewritten.
- Every moving cutout is fully opaque inside the mask: people, animals, bikes, deliveries, and vehicles. Only a 2 to 3 pixel edge is soft, and the fade in and out is 0.2 seconds. The background plate is dimmed and desaturated by about 18% so the objects read first.
- The cutout window is the union of the detector boxes in that clip, plus about 12% on the sides. People and animals also get about 20% above the box, and a vehicle gets extra room in the direction it is moving, so a nose or a head is not cropped off. The outline follows a closed, simplified contour rather than every jagged pixel.
- Each time label is drawn once. The player does not paint the same words again on top of the frame.
- A label and its leader are drawn only when the cutout is on screen and readable against the plate. The leader ends on the centroid of the mask that was painted. An empty mask, a box that sits mostly off the frame, a cutout under about 0.15% of the frame, or one with almost no contrast against the plate does not get a line.

Tap a time on the video, or a row in the event list, to pause and open that clip. The dialog has download, previous, and next, and a link to Frigate review at that time. While the synopsis is paused, every time on screen can be tapped. Live clips play from the camera's HLS VOD playlist (`/vod/<camera>/start/<ts>/end/<ts>/index.m3u8`), which Safari plays natively. Archived files are served as a faststart H.264 MP4 with range requests. If Frigate has already deleted the recording, and `recap.archive` is set, the player uses the archived file for the same event.

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

## Rolling recap

Each recap camera keeps one rolling recap of the last `rolling_hours` (default 6) ready, so the Recap page can play it right away. Every `rolling_interval_minutes` (default 30, `0` turns it off) Frigate checks the events in that window. If no event is new and none changed its end time, the refresh is skipped. Otherwise the recap is rebuilt in a staging folder and swapped into place, so there is only ever one live rolling recap per camera and the previous one keeps playing until the new one is ready. If a refresh fails or is cancelled, the last good one stays. The rolling recap it replaces is moved into the saved list (marked as an earlier rolling recap), unless it held exactly the same events. Saved copies are removed after `rolling_keep_hours` (default 72, `0` deletes them on replace). Nightly, custom, and manual recaps are never removed by this. After `rolling_max_age_minutes` (default 180) it is rebuilt even with no new events, so objects that aged out of the window drop off.

```yaml
recap:
  rolling_hours: 6
  rolling_interval_minutes: 30
  rolling_max_age_minutes: 180
  rolling_keep_hours: 72
```

Queue rules: one recap is built at a time. A request from the UI (a longer range, a custom range, or Refresh now) runs before scheduled work. A refresh never waits behind another refresh of the same camera, and a request that is already waiting or running is returned instead of being queued twice.

Each event's cutouts are cached under `/media/frigate/recap/.cutcache`, keyed on the event id and end time, output size, sample rate, the per-object time cap, the category, and a cache version. A refresh only decodes events it has not seen. Entries unused for 74 hours are removed after each job, and the size is written to the log. Manifests record `started`, `finished`, `took_s`, the start of each stage, and cache hits and misses.

`interval_minutes` is a separate optional job, off unless you set it. It builds a new recap of only the last that many minutes, with kind `rolling-<minutes>m`. Saved copies of that kind are replaced when the new one is Ready. Each object is still trimmed to `max_object_seconds` of samples, so a 30 minute run is a small fraction of a nightly 24 hour recap. The finished-recap log includes `rendered in` seconds (`took_s` on the manifest). If that render is still going when the next interval opens, that run is skipped instead of queued.

`GET /api/recap/rolling/<camera>` returns the rolling recap, any refresh in progress, and the last outcome. `POST /api/recap/rolling/<camera>/refresh` checks now (`?force=true` rebuilds even when nothing changed).

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
- At most `max_events` events are processed (default 400). When a window has more, the newest are kept, after parked vehicles are set aside.
- An archive URL alone cannot join multiple recording files. Mount `archive.path` for that.
