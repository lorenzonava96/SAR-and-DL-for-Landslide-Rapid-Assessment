#!/usr/bin/env python3
"""Download and verify SAR-LRA model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="model/weights-manifest.json")
    parser.add_argument("--directory", default="model/weights")
    parser.add_argument("--update-manifest", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    destination = Path(args.directory)
    destination.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = 0

    for item in manifest["weights"]:
        target = destination / item["filename"]
        if not target.exists():
            print(f"Downloading {item['orbit']} weights...")
            with urlopen(item["source_url"]) as response, target.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)

        actual = sha256_file(target)
        expected = item.get("sha256")

        if expected in (None, "", "TO_BE_VERIFIED"):
            print(f"{item['orbit']}: sha256={actual}")
            if args.update_manifest:
                item["sha256"] = actual
        elif actual != expected:
            failures += 1
            print(f"{item['orbit']}: CHECKSUM MISMATCH expected={expected} actual={actual}")
        else:
            print(f"{item['orbit']}: checksum verified")

    if args.update_manifest:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {manifest_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
