#!/usr/bin/env python3
"""Patch ALFWorld game.tw-pddl grammar: replace lend:location -> l:location.

Fixes tatsu parser errors (FailedToken / FailedCut) when loading games.
Run once from repo root: python scripts/alfworld/patch_alfworld_grammar.py
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALFWORLD_DATA = REPO_ROOT / "data" / "alfworld"

BAD = "lend:location"
GOOD = "l:location"


def main() -> None:
    files = list(ALFWORLD_DATA.rglob("game.tw-pddl"))
    patched = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Skip (read error) {path}: {e}")
            continue
        if BAD not in text:
            continue
        try:
            # Replace in raw text so JSON structure is preserved
            new_text = text.replace(BAD, GOOD)
            path.write_text(new_text, encoding="utf-8")
            patched += 1
        except Exception as e:
            print(f"Skip (write error) {path}: {e}")
    print(f"Patched {patched} of {len(files)} game.tw-pddl files (replaced '{BAD}' -> '{GOOD}').")


if __name__ == "__main__":
    main()
