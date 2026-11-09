#!/bin/bash
# Deploy the Recap build of Frigate 0.18.2.
#
# Usage: ./deploy-0.18.2.sh <backup-dir> [image]
#   backup-dir  Host path that receives config, compose, and the database.
#               The LXC root has about 10 GB free, so pass a directory on the
#               Proxmox host (for example a bind mount of /var/lib/vz). This
#               script never deletes files to make room, and it never removes
#               the copies it writes.
#   image       Local tag or GHCR ref to run. Default: frigate:recap-0.18.2
#               when that tag exists, otherwise ghcr.io/sampsonb/frigate:recap-0.18.2.
#
# The compose file keeps its volumes, devices, and ports. Only the image line
# changes, to frigate:recap-0.18.2. frigate:recap-0.17.2 is left in place so a
# failed boot can roll back to it. 0.18 applies database migrations 033-035,
# which 0.17 cannot read, so a failed boot also restores the pre-swap database
# (the db file and its WAL and shm).
#
# Modeled on deploy-c55234a.sh. Expects /root/frigate-recap-deploy/common.sh.
set -uo pipefail

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 <backup-dir> [image]" >&2
  exit 1
fi

BACKUP_DIR=$1
shift || true
source /root/frigate-recap-deploy/common.sh

if docker image inspect frigate:recap-0.18.2 >/dev/null 2>&1; then
  DEFAULT_IMAGE=frigate:recap-0.18.2
else
  DEFAULT_IMAGE=ghcr.io/sampsonb/frigate:recap-0.18.2
fi
NEW=${1:-$DEFAULT_IMAGE}
LIVE_TAG=frigate:recap-0.17.2
NEW_TAG=frigate:recap-0.18.2
# A second full Frigate image is about 8 GB. Below this, docker pulls or
# unpacks can fill the LXC and stop recordings. Nothing is deleted to get
# under this line.
IMAGE_FREE_GB=8
# Room for the database (about 0.2 GB today), its WAL, and the config copies.
BACKUP_FREE_GB=2

log "=== deploy $NEW as $NEW_TAG (rollback target $LIVE_TAG)"
log "backups -> $BACKUP_DIR"

docker image inspect "$LIVE_TAG" >/dev/null || { log "ABORT: $LIVE_TAG missing"; exit 1; }
mkdir -p "$BACKUP_DIR" || { log "ABORT: cannot create $BACKUP_DIR"; exit 1; }
[[ -d "$BACKUP_DIR" && -w "$BACKUP_DIR" ]] || { log "ABORT: $BACKUP_DIR is not a writable directory"; exit 1; }

free_gb() {
  df --output=avail -BG "$1" | tail -1 | tr -dc 0-9
}
root_free=$(free_gb /)
backup_free=$(free_gb "$BACKUP_DIR")
log "free space: / has ${root_free}G, backup dir has ${backup_free}G"
if (( root_free < IMAGE_FREE_GB )); then
  log "ABORT: only ${root_free}G free on /. The 0.18.2 image needs about ${IMAGE_FREE_GB}G and this script will not delete anything to free it."
  exit 1
fi
if (( backup_free < BACKUP_FREE_GB )); then
  log "ABORT: only ${backup_free}G free in $BACKUP_DIR. Need ${BACKUP_FREE_GB}G for the database, WAL, config, and compose. Nothing was deleted."
  exit 1
fi

if [[ "$NEW" != */* ]] && docker image inspect "$NEW" >/dev/null 2>&1; then
  log "using local image $NEW id=$(docker image inspect "$NEW" --format '{{.Id}}')"
else
  log "pulling $NEW"
  if ! docker pull "$NEW"; then
    log "ABORT: pull failed. Nothing was changed."
    exit 1
  fi
fi
docker tag "$NEW" "$NEW_TAG"

VER=$(docker run --rm --entrypoint cat "$NEW_TAG" /opt/frigate/frigate/version.py | tr -d '\n')
log "image version: $VER"
[[ "$VER" == *0.18.2* ]] || { log "ABORT: not 0.18.2 ($VER)"; exit 1; }
docker run --rm --entrypoint test "$NEW_TAG" -f /opt/frigate/frigate/api/recap.py || { log "ABORT: no recap API"; exit 1; }
docker run --rm --entrypoint test "$NEW_TAG" -f /openvino-model/ssdlite_mobilenet_v2.xml || { log "ABORT: OpenVINO model missing"; exit 1; }
docker run --rm --entrypoint sh "$NEW_TAG" -c 'grep -lq recapPlaybackRate /opt/frigate/web/assets/Recap-*.js' || { log "ABORT: image has no speed buttons"; exit 1; }
grep -qE '^\s+image:\s*' "$COMPOSE" || { log "ABORT: no image line in compose"; exit 1; }

# Exact pre-swap copies. sqlite .backup folds the WAL into one consistent
# file. The raw db, WAL, and shm are kept beside it so the rollback can put
# back the files Frigate had open.
SNAP=$BACKUP_DIR/pre-0.18.2-$TS
mkdir -p "$SNAP"
cp -a "$CONFIG" "$SNAP/config.yml"
cp -a "$COMPOSE" "$SNAP/docker-compose.yml"
if [[ -f "$DB" ]]; then
  cp -a "$DB" "$SNAP/frigate.db"
  nice -n 19 sqlite3 "$DB" ".timeout 20000" ".backup $SNAP/frigate.db.consistent" \
    && log "consistent db snapshot $SNAP/frigate.db.consistent"
fi
if [[ -f "$DB-wal" ]]; then
  cp -a "$DB-wal" "$SNAP/frigate.db-wal"
  log "saved WAL $(stat -c %s "$SNAP/frigate.db-wal") bytes"
fi
if [[ -f "$DB-shm" ]]; then
  cp -a "$DB-shm" "$SNAP/frigate.db-shm"
fi
log "pre-swap copies in $SNAP (left in place; this script does not remove them)"

# Point compose at the new tag without touching the 0.17.2 tag.
cp -a "$COMPOSE" "$SNAP/docker-compose.yml.before-edit"
sed -i -E "s|^(\\s+image:\\s*).*$|\\1${NEW_TAG}|" "$COMPOSE"
log "compose image -> $(grep -E '^\\s+image:' "$COMPOSE")"
compose config -q || { log "ABORT: compose invalid"; cp -a "$SNAP/docker-compose.yml" "$COMPOSE"; exit 1; }

restore_db() {
  log "restoring pre-swap database from $SNAP"
  compose stop frigate || true
  # Move the live generation into the backup dir so a 0.18 WAL cannot replay
  # onto the restored 0.17 file. Those files are kept. Nothing is deleted.
  moved=$SNAP/replaced-by-rollback
  mkdir -p "$moved"
  [[ -f "$DB" ]] && mv -f "$DB" "$moved/frigate.db"
  [[ -f "$DB-wal" ]] && mv -f "$DB-wal" "$moved/frigate.db-wal"
  [[ -f "$DB-shm" ]] && mv -f "$DB-shm" "$moved/frigate.db-shm"
  if [[ -f "$SNAP/frigate.db.consistent" ]]; then
    cp -a "$SNAP/frigate.db.consistent" "$DB"
    log "restored consistent snapshot (WAL contents included); live files kept in $moved"
  elif [[ -f "$SNAP/frigate.db" ]]; then
    cp -a "$SNAP/frigate.db" "$DB"
    [[ -f "$SNAP/frigate.db-wal" ]] && cp -a "$SNAP/frigate.db-wal" "$DB-wal"
    [[ -f "$SNAP/frigate.db-shm" ]] && cp -a "$SNAP/frigate.db-shm" "$DB-shm"
    log "restored raw db and WAL; previous live files kept in $moved"
  else
    log "WARNING: no database copy in $SNAP"
  fi
}

rollback() {
  log "!!! deploy unhealthy -> rolling back to $LIVE_TAG and the pre-swap database"
  docker logs --since 10m frigate 2>&1 | tail -40 || true
  cp -a "$SNAP/docker-compose.yml" "$COMPOSE"
  cp -a "$SNAP/config.yml" "$CONFIG"
  restore_db
  docker image inspect "$LIVE_TAG" >/dev/null || { log "FATAL: $LIVE_TAG disappeared"; exit 1; }
  compose up -d --pull never --force-recreate frigate
  if wait_api 300 >/dev/null; then
    log "rolled back, API up ($(curl -fs --max-time 3 "$API/api/version" || true))"
  else
    log "rolled back, API not up yet"
  fi
}

compose up -d --pull never --force-recreate frigate
if V=$(wait_api 300); then log "API up: $V"; else log "API did not come up"; V=""; fi
recordings_ok() {
  curl -fs --max-time 10 "$API/api/recordings/summary" | python3 -c '
import json, sys
body = json.load(sys.stdin)
if not isinstance(body, dict) or not body:
    sys.exit("recordings summary empty")
print("  recordings days:", len(body))
'
}

if [[ -n "$V" ]] && [[ "$V" == *0.18.2* ]] && sleep 20 && check_cameras && check_restarts && recordings_ok && curl -fs --max-time 5 "$API/api/recap" >/dev/null; then
  log "=== DEPLOY OK: $NEW_TAG ($VER)"
  log "rollback copies remain in $SNAP"
  exit 0
fi
rollback
exit 2
