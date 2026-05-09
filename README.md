# MusicGeneration-Lab1

Symbolic piano music continuation for binary piano-roll data. The dataset is not
audio: every sequence is a matrix `[T, 88]`, where each row is an onset frame,
columns are MIDI notes 21 through 108, and `step_sec` is usually `0.05`.

## Model

The main model is a Nested Music Transformer-inspired piano-roll Transformer.
Each time step is treated as one compound musical token containing 88 binary note
attributes. A causal temporal Transformer models previous frames, then a note
head predicts the next 88 onset attributes.

This is adapted directly to the teacher's native `[T, 88]` binary piano-roll
format. It does not convert to audio tokens, MIDI event tokens, GRU, LSTM, or a
traditional ML model.

Architecture:

- Input windows: `[B, context_len, 88]`
- Frame embedding: linear projection `88 -> d_model`
- Learned positional embeddings
- Causal Transformer encoder stack
- Output logits: `[B, context_len, 88]`
- Loss: `BCEWithLogitsLoss` against the next frame
- Head modes: stable default `mlp`, optional `nested` pitch-conditioned head

## Installation

Use Python 3.10+ if possible.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy torch
```

If you want notebook playback or extra analysis tools, install them separately.
The training/generation scripts only require NumPy and PyTorch.

## Inspect Data

```bash
python inspect_npz.py --npz train.npz
python inspect_npz.py --npz eval_set_01_prefix.npz
```

## Prepare Cache

`train.npz` decompresses to a large array, so training uses a one-time `.npy`
cache that can be memory-mapped.

```bash
python prepare_cache.py --train_npz train.npz --cache_dir cache
```

This creates:

- `cache/rolls_flat.npy`
- `cache/offsets.npy`
- `cache/ids.npy`
- `cache/dataset_info.json`

If the cache already exists, training reuses it with `np.load(...,
mmap_mode="r")`.

## Train

```bash
python train.py --train_npz train.npz --cache_dir cache --context_len 512 --batch_size 16 --epochs 20 --amp
```

Equivalent full form:

```bash
python train.py \
  --train_npz train.npz \
  --cache_dir cache \
  --context_len 512 \
  --batch_size 16 \
  --epochs 20 \
  --steps_per_epoch 1000 \
  --lr 3e-4 \
  --d_model 256 \
  --n_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --out_dir checkpoints \
  --device auto \
  --pos_weight auto \
  --amp
```

Training uses random sequence windows of length `context_len + 1`. The input is
`window[:-1]`; the target is `window[1:]`. Validation is split by sequence ID,
not by random frames, to reduce leakage.

Checkpoints:

- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- `checkpoints/config.json`

Resume:

```bash
python train.py --train_npz train.npz --cache_dir cache --resume checkpoints/latest.pt --amp
```

## Generate Continuations

The generator accepts any prefix `.npz` with the same format. It does not depend
on `eval_set_01`.

```bash
python generate.py --checkpoint checkpoints/best.pt --prefix_npz eval_set_01_prefix.npz --out_npz outputs/eval_set_01_generated.npz --continuation_steps 2048
```

To match a known full-file length for local testing only:

```bash
python generate.py \
  --checkpoint checkpoints/best.pt \
  --prefix_npz eval_set_01_prefix.npz \
  --out_npz outputs/eval_set_01_generated_matched.npz \
  --target_full_npz eval_set_01_full.npz \
  --match_full_lengths
```

During the real presentation, use only the prefix file and choose
`--continuation_steps`.

Generation guarantees:

- The prefix frames are copied unchanged.
- Continuation is autoregressive, one frame at a time.
- Output `rolls_flat` is binary `uint8` with shape `[total_T, 88]`.
- The output includes `rolls_flat`, `offsets`, `ids`, metadata fields,
  `is_prefix=False`, `prefix_steps`, and `generated_steps`.

Useful controls:

- `--temperature`: probability sharpness
- `--note_threshold`: suppress very low-probability notes
- `--top_k_notes`: cap candidate notes per onset frame
- `--max_notes_per_event`: avoid dense all-note bursts
- `--min_notes_per_event`: force notes if desired, default allows silence
- `--seed`: reproducible sampling
- `--repetition_patience`: reduce exact repeated non-silent frames

## Preview Audio

Use the provided `audio_play.py` synthesis helper:

```bash
python preview_audio.py --npz outputs/eval_set_01_generated.npz --index 0 --out_wav outputs/generated_0.wav
```

Optional:

```bash
python preview_audio.py --npz outputs/eval_set_01_generated.npz --index 0 --out_wav outputs/sample0.wav --num_steps 2048
```

## Files

- `inspect_npz.py`: prints keys, shapes, offsets, sequence lengths, and metadata.
- `prepare_cache.py`: converts `train.npz` to memory-mapped `.npy` cache files.
- `dataset.py`: random fixed-length training windows with sequence-level split.
- `model.py`: causal `NestedPianoRollTransformer`.
- `train.py`: training loop with positive weighting, AMP, clipping, checkpoints,
  config saving, and resume.
- `generate.py`: autoregressive continuation for arbitrary prefix `.npz` files.
- `preview_audio.py`: exports generated sequences to WAV via `audio_play.py`.
