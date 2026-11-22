"""Validate a local ASVspoof 2019 LA extraction before training."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .protocol import EXPECTED_COUNTS, load_protocol


def validate_dataset(la_root: str | Path, check_audio: bool = True) -> bool:
    """Print split summaries and return whether layout/counts/files are valid."""
    root = Path(la_root)
    valid = True
    print(f"LA root: {root.resolve()}")

    for split, expected in EXPECTED_COUNTS.items():
        try:
            entries = load_protocol(root, split)
        except (OSError, ValueError) as exc:
            print(f"[{split}] ERROR: {exc}")
            valid = False
            continue

        counts = Counter(entry.label for entry in entries)
        missing = []
        if check_audio:
            missing = [entry.audio_path for entry in entries if not entry.audio_path.is_file()]
        attacks = sorted({entry.attack_id for entry in entries if entry.attack_id != "-"})
        matches = dict(counts) == expected
        valid &= matches and not missing
        print(
            f"[{split}] total={len(entries)} bonafide={counts['bonafide']} "
            f"spoof={counts['spoof']} attacks={','.join(attacks)}"
        )
        if not matches:
            print(f"[{split}] ERROR: expected counts {expected}")
        if missing:
            print(f"[{split}] ERROR: {len(missing)} missing audio files; first={missing[0]}")

    print("Dataset validation PASSED" if valid else "Dataset validation FAILED")
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--la-root", required=True, type=Path)
    parser.add_argument(
        "--skip-audio-check",
        action="store_true",
        help="Check protocols/counts only (faster, less complete).",
    )
    args = parser.parse_args()
    raise SystemExit(not validate_dataset(args.la_root, not args.skip_audio_check))


if __name__ == "__main__":
    main()
