# Event-State Music Transformer

PyTorch project for symbolic music continuation from piano-roll `.npz` files.

## Files

- `vocab.py`: builds and stores the top-K chord-state vocabulary.
- `dataset.py`: converts rolls to event-state sequences and creates padded random chunks.
- `model.py`: decoder-only Event-State Music Transformer.
- `train.py`: trains with AdamW, warmup/cosine schedule, AMP, clipping, and best-checkpoint saving.
- `generate.py`: samples continuations from `eval_set_01_prefix.npz`.
- `generate_wav.py`: exports generated `.npz` sequences to WAV with `audio_play.py`.
- `inspect_dataset.py`: prints dataset and event statistics.

## Quick Start

```bash
cd event_state_transformer
python3 train.py --data-dir .. --out-dir runs/event_state
python3 generate.py --data-dir .. --checkpoint runs/event_state/best.pt --out runs/event_state/eval_set_01_generated_event_state.npz
python3 generate_wav.py --npz runs/event_state/eval_set_01_generated_event_state.npz --out-dir runs/event_state/wav
```

Resume training:

```bash
python3 train.py --data-dir .. --out-dir runs/event_state --resume runs/event_state/last.pt --epochs 60
```

For older model-only checkpoints, use the same command; the script will load weights and restart optimizer/scheduler state. Add `--resume-model-only` to intentionally do that with a full checkpoint.

The implementation uses event pairs rather than raw frame prediction. Generation uses temperature and nucleus sampling; it does not use argmax.
