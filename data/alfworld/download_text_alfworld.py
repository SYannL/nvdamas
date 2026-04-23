#!/usr/bin/env python3
"""Download the text-only ALFWorld assets needed by this project.

This script only downloads the textual resources:
- ALFWorld JSON files
- PDDL files
- TextWorld game files (`game.tw-pddl`)
- Logic files (`alfred.pddl`, `alfred.twl2`)

It intentionally skips visual assets such as detectors and checkpoints.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent

ARCHIVES = {
    "json_2.1.1_json.zip": "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip",
    "json_2.1.1_pddl.zip": "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip",
    "json_2.1.1_tw-pddl.zip": "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_tw-pddl.zip",
}

LOGIC_FILES = {
    "logic/alfred.pddl": "https://raw.githubusercontent.com/alfworld/alfworld/master/alfworld/data/alfred.pddl",
    "logic/alfred.twl2": "https://raw.githubusercontent.com/alfworld/alfworld/master/alfworld/data/alfred.twl2",
}

EXPECTED_PATHS = (
    "json_2.1.1/train",
    "json_2.1.1/valid_seen",
    "json_2.1.1/valid_unseen",
    "logic/alfred.pddl",
    "logic/alfred.twl2",
)


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{num_bytes}B"


def print_progress(prefix: str, downloaded: int, total: int | None) -> None:
    if total:
        percent = downloaded / total * 100
        message = f"\r{prefix}: {percent:6.2f}% ({format_bytes(downloaded)}/{format_bytes(total)})"
    else:
        message = f"\r{prefix}: {format_bytes(downloaded)}"
    print(message, end="", flush=True)


def download(url: str, destination: Path, force_download: bool) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force_download:
        print(f"Skip existing archive: {destination}")
        return destination

    tmp_dir = Path(tempfile.mkdtemp(prefix="alfworld_download_"))
    tmp_path = tmp_dir / destination.name
    print(f"Downloading {url}")

    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total is not None else None
            downloaded = 0

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                print_progress(destination.name, downloaded, total_bytes)

        print()
        shutil.move(str(tmp_path), str(destination))
        return destination
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def extract_zip(zip_path: Path, output_dir: Path, force: bool) -> None:
    print(f"Extracting {zip_path.name} -> {output_dir}")
    skipped = 0

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target_path = output_dir / member.filename
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            if target_path.exists() and not force:
                skipped += 1
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    if skipped:
        print(f"Skipped {skipped} existing files. Use --force to overwrite extracted files.")


def download_logic_files(output_dir: Path, force: bool) -> None:
    for relative_path, url in LOGIC_FILES.items():
        target = output_dir / relative_path
        if target.exists() and not force:
            print(f"Skip existing logic file: {target}")
            continue

        print(f"Downloading {url} -> {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response, target.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def validate_layout(output_dir: Path) -> list[str]:
    missing = []
    for relative_path in EXPECTED_PATHS:
        if not (output_dir / relative_path).exists():
            missing.append(relative_path)
    return missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the text-only ALFWorld dataset and logic files."
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT),
        help="Target directory for the ALFWorld text assets. Defaults to this folder.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite extracted files and logic files if they already exist.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download archives even if the zip files already exist locally.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the downloaded zip files after extraction.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_dir = output_dir / "_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    try:
        for archive_name, url in ARCHIVES.items():
            archive_path = download(url, archive_dir / archive_name, args.force_download)
            extract_zip(archive_path, output_dir, args.force)
            if not args.keep_archives:
                archive_path.unlink(missing_ok=True)

        download_logic_files(output_dir, args.force)
    except Exception as exc:  # pragma: no cover - defensive CLI error handling
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    if not args.keep_archives and archive_dir.exists() and not any(archive_dir.iterdir()):
        archive_dir.rmdir()

    missing = validate_layout(output_dir)
    if missing:
        print("Finished, but some expected paths are still missing:", file=sys.stderr)
        for relative_path in missing:
            print(f"  - {relative_path}", file=sys.stderr)
        return 2

    print("ALFWorld text-only assets are ready.")
    print(f"Dataset root: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
