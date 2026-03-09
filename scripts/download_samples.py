"""Download official Qwen3-ASR sample audio files for local testing."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen


SAMPLES = {
    "asr_zh.wav": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
    "asr_en.wav": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official Qwen3-ASR sample audio into samples/."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    samples_dir = root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in SAMPLES.items():
        destination = samples_dir / filename
        if destination.exists() and not args.force:
            print(f"skip {destination} (already exists)")
            continue

        print(f"download {url} -> {destination}")
        with urlopen(url) as response:
            data = response.read()
        destination.write_bytes(data)
        print(f"saved {destination} ({len(data)} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
