"""Download official Qwen3-ASR sample audio files and target text."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen


SAMPLES = [
    {
        "filename": "asr_zh.wav",
        "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-ASR/opensource/audios/fast1.wav",
        "target": "蹦出来之后，左手、右手接一个慢动作，右边再直接拉到这上面之后，直接拉到这个轮胎上，上边再接过去之后，然后上边再直接拉到这个位置了之后，右边再直接这个位置接倒过去的之后，再倒一下，然后右边再直接抓住这个上边了之后，直接从这边上边过去了之后，直接抓住这个树杈，然后这个位置直接倒到这个树杈。",
    },
    {
        "filename": "asr_en.wav",
        "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-ASR/opensource/audios/f1_noise.wav",
        "target": "Okay, Charles. It looks like we have a problem with the radio. What happened? Yeah, someone spilled water on their machine. I uh, yeah. Charles, can you hear us? Mamma mia.",
    },
    {
        "filename": "asr_yue.wav",
        "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-ASR/opensource/audios/cantonese.wav",
        "target": "今次寻寻觅觅，终于揾到my princess，肯借个场俾我哋玩。你知啦，喺香港地喺繁忙时间要揾个场嚟拍嘢系非常之难嘅。再一次多谢你哋，亦都好多谢片入边嘅每一个人。下一次我哋斗啲咩好？喺下面留言话我哋知啦。拜拜。",
    },
    {
        "filename": "asr_multi.wav",
        "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-ASR/opensource/audios/codeswitch.wav",
        "target": "I'm alone, all by myself. Je suis tout seul. Sono tutto. Estoy solo.",
    },
    {
        "filename": "asr_ar.wav",
        "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-ASR/opensource/audios/multilingual/ar1.wav",
        "target": "إطلالات مكياج عيون ذهبي لسهرات صيف عشرين واحد وعشرين بأسلوب النجمات.",
    },
    {
        "filename": "asr_de.wav",
        "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-ASR/opensource/audios/multilingual/de.wav",
        "target": "Raptorium Bergbau scheint profitierter als Monroe als Reaktion auf die wirtschaftlichen Ausfälle zu sein.",
    },
    {
        "filename": "asr_es.wav",
        "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-ASR/opensource/audios/multilingual/es1.wav",
        "target": "Esta prenda es amplia, recomiendo elegir una talla menor a la habitual.",
    },
    {
        "filename": "asr_fr.wav",
        "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-ASR/opensource/audios/multilingual/fr1.wav",
        "target": "Alice et moi sommes allés à Paris voyager en train au printemps, c'était très amusant.",
    },
    {
        "filename": "asr_ru.wav",
        "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-ASR/opensource/audios/multilingual/ru1.wav",
        "target": "Барсук, живущий в киевском зоопарке, совершил побег из своего вольера.",
    },
    {
        "filename": "asr_ja.wav",
        "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-ASR/opensource/audios/multilingual/ja1.wav",
        "target": "抜群の運動神経を持ち合わせ、どんな要求にも応えてきた。",
    },
]


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

    for sample in SAMPLES:
        audio_path = samples_dir / sample["filename"]
        target_path = audio_path.with_suffix(".target.txt")

        if audio_path.exists() and not args.force:
            print(f"skip {audio_path} (already exists)")
        else:
            print(f"download {sample['url']} -> {audio_path}")
            with urlopen(sample["url"]) as response:
                data = response.read()
            audio_path.write_bytes(data)
            print(f"saved {audio_path} ({len(data)} bytes)")

        target_path.write_text(sample["target"] + "\n", encoding="utf-8")
        print(f"saved {target_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
