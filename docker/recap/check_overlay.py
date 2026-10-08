"""Fail if the recap overlay image would miss a changed runtime file.

Compares this branch with recap-base-0.17.2. Python files under frigate/
(except tests) must be named in docker/recap/Dockerfile. Everything under
web/ is covered by the rebuilt dist copy. Docs, tests, and this docker
directory are not part of the running container.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker" / "recap" / "Dockerfile"
BASE_CANDIDATES = (
    "origin/recap-base-0.17.2",
    "recap-base-0.17.2",
)


def _base_ref() -> str:
    for ref in BASE_CANDIDATES:
        found = subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if found.returncode == 0:
            return ref
    raise SystemExit(
        "Cannot find recap-base-0.17.2. Fetch it before running this check."
    )


def changed_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{_base_ref()}...HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def covered(path: str, dockerfile: str) -> bool:
    if path.startswith("web/"):
        return "COPY --from=web /work/dist/ /opt/frigate/web/" in dockerfile
    if path.startswith("frigate/test/"):
        return True
    if not path.startswith("frigate/"):
        return True
    if path.startswith("frigate/recap/"):
        return "COPY frigate/recap/ /opt/frigate/frigate/recap/" in dockerfile
    needle = f"COPY {path} "
    return needle in dockerfile


def main() -> int:
    dockerfile = DOCKERFILE.read_text()
    missing = [path for path in changed_files() if not covered(path, dockerfile)]
    if missing:
        print("docker/recap/Dockerfile does not overlay these runtime files:")
        for path in missing:
            print(f"  {path}")
        return 1
    print("recap overlay covers every changed runtime file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
