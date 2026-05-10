from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a generated .npz sequence to WAV using project audio_play.py.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out_wav", type=Path, required=True)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=2048)
    parser.add_argument("--sample_rate", type=int, default=22050)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from audio_play import export_dataset_sequence_wav

    out = export_dataset_sequence_wav(
        npz_path=args.npz,
        index=args.index,
        out_path=args.out_wav,
        start_step=args.start_step,
        num_steps=args.num_steps,
        sample_rate=args.sample_rate,
    )
    print(f"saved wav: {out}")


if __name__ == "__main__":
    main()
