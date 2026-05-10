# MusicGeneration-Lab1

Symbolic piano continuation for binary onset piano-roll data. The dataset is not
audio: each sequence is a `[T, 88]` matrix, where columns are MIDI notes 21-108,
values are `0/1`, `representation="onset"`, and `step_sec=0.05`.

## Recommended Model

Use the token-based chord Transformer pipeline:

- Non-silent onset frames become `CHORD_x` tokens.
- Silent gaps become `TIME_SHIFT_n` tokens.
- A causal Transformer predicts the next symbolic event token.
- Generated event tokens are decoded back to the teacher's required `[T, 88]`
  piano-roll format.

This replaces the earlier main approach that predicted 88 independent BCE logits
per frame. That frame-BCE version can still be useful as a baseline, but it has a
bad sampling tradeoff for sparse onset data: a high threshold can produce total
silence, while a very low threshold and high temperature can produce random note
noise. Tokenizing observed chord/onset patterns makes the output space musical:
the model predicts valid observed events rather than 88 unrelated yes/no notes.

The design is Nested Music Transformer-inspired because each musical event is a
compound/chord event, the temporal Transformer models event sequences, and the
decoder emits valid symbolic music events. It is adapted to the native `[T, 88]`
piano-roll data and does not use GRU, LSTM, audio models, or the old Music
Transformer implementation as the main model.

We do not enumerate all `2^88` possible note combinations. The chord vocabulary
is built only from observed nonzero frames in `train.npz`, plus all 88 single-note
chords as a robust fallback. Unknown chords map to `UNK_CHORD`; decoding
`UNK_CHORD` produces silence.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy torch
```

## Inspect Data

```bash
python inspect_npz.py --npz train.npz
python inspect_npz.py --npz eval_set_01_prefix.npz
```

## Prepare Token Cache

```bash
python prepare_token_cache.py \
  --train_npz train.npz \
  --cache_dir token_cache \
  --max_shift 64 \
  --min_chord_freq 1 \
  --max_chord_vocab 20000
```

This saves:

- `token_cache/tokenizer.json`
- `token_cache/tokens_flat.npy`
- `token_cache/token_offsets.npy`
- `token_cache/ids.npy`
- `token_cache/dataset_info.json`

`TIME_SHIFT_1 ... TIME_SHIFT_64` compress sparse silence. Longer silent gaps are
split into multiple max-shift tokens.

## Train Token Model

```bash
python train_token_model.py \
  --cache_dir token_cache \
  --out_dir checkpoints_token \
  --context_len 1024 \
  --batch_size 8 \
  --epochs 20 \
  --steps_per_epoch 1000 \
  --lr 3e-4 \
  --d_model 384 \
  --n_layers 8 \
  --n_heads 8 \
  --dropout 0.1 \
  --amp
```

Training uses token-level `CrossEntropyLoss(ignore_index=PAD)`, AdamW, gradient
clipping, sequence-level validation split, CUDA/CPU auto-detection, optional AMP,
and checkpoint resume.

Resume:

```bash
python train_token_model.py \
  --cache_dir token_cache \
  --out_dir checkpoints_token \
  --context_len 1024 \
  --batch_size 8 \
  --epochs 20 \
  --resume checkpoints_token/latest.pt \
  --amp
```

Checkpoints:

- `checkpoints_token/latest.pt`
- `checkpoints_token/best.pt`
- `checkpoints_token/config.json`

## Generate From Any Prefix NPZ

Mode A, prefix length plus continuation:

```bash
python generate_token.py \
  --checkpoint checkpoints_token/best.pt \
  --tokenizer token_cache/tokenizer.json \
  --prefix_npz eval_set_01_prefix.npz \
  --out_npz outputs/eval_set_01_generated_token.npz \
  --continuation_steps 2048 \
  --context_len 1024 \
  --temperature 0.85 \
  --top_k 32 \
  --top_p 0.95 \
  --repetition_penalty 1.1
```

Mode B, exact total length for every sequence:

```bash
python generate_token.py \
  --checkpoint checkpoints_token/best.pt \
  --tokenizer token_cache/tokenizer.json \
  --prefix_npz eval_set_01_prefix.npz \
  --out_npz outputs/generated_total_4096.npz \
  --target_total_steps 4096
```

Mode C, match a full file for local testing only:

```bash
python generate_token.py \
  --checkpoint checkpoints_token/best.pt \
  --tokenizer token_cache/tokenizer.json \
  --prefix_npz eval_set_01_prefix.npz \
  --out_npz outputs/eval_set_01_generated_token_matched.npz \
  --target_full_npz eval_set_01_full.npz \
  --match_full_lengths
```

The teacher may provide a different prefix file during evaluation. Do not depend
on `eval_set_01`; pass the provided prefix path to `--prefix_npz`.

Generation behavior:

- Encodes the prefix roll into event tokens.
- Samples event tokens autoregressively using temperature, top-k, and top-p.
- Applies simple density and repetition controls.
- Decodes tokens back to `[T, 88]`.
- Restores the original prefix frames exactly.
- Pads/truncates to the requested target frame length.
- Saves a `rolls_flat + offsets` NPZ compatible with `audio_play.py`.

Output fields include:

- `rolls_flat`
- `offsets`
- `ids`
- `step_sec`
- `note_min`
- `note_max`
- `num_positions`
- `representation`
- `is_prefix=False`
- `prefix_steps`
- `generated_steps`
- `model_type="NestedMusicTransformerInspired_TokenChordTransformer"`

## Analyze Generation

```bash
python analyze_generation.py \
  --npz outputs/eval_set_01_generated_token.npz \
  --prefix_npz eval_set_01_prefix.npz
```

With training density warning:

```bash
python analyze_generation.py \
  --npz outputs/eval_set_01_generated_token.npz \
  --prefix_npz eval_set_01_prefix.npz \
  --tokenizer token_cache/tokenizer.json
```

The analyzer prints generated note counts, event density, unique chord patterns,
longest silent run, top generated chords, and warnings for silence or excessive
density.

## Preview Audio

```bash
python preview_audio.py \
  --npz outputs/eval_set_01_generated_token.npz \
  --index 0 \
  --out_wav outputs/generated_token_0.wav \
  --num_steps 2048
```

## Legacy Frame-BCE Baseline

The older frame model remains available:

```bash
python prepare_cache.py --train_npz train.npz --cache_dir cache
python train.py --train_npz train.npz --cache_dir cache --context_len 512 --batch_size 16 --epochs 20 --amp
python generate.py --checkpoint checkpoints/best.pt --prefix_npz eval_set_01_prefix.npz --out_npz outputs/eval_set_01_generated.npz
```

For the final project, prefer the token model because it models sparse symbolic
events directly.

## Files

- `tokenizer.py`: `PianoEventTokenizer` with observed chord vocabulary and
  `TIME_SHIFT` compression.
- `prepare_token_cache.py`: builds tokenizer and tokenized train cache.
- `token_dataset.py`: token windows with sequence-level validation split.
- `token_model.py`: causal token Transformer.
- `train_token_model.py`: CrossEntropy token-model training.
- `generate_token.py`: event-token autoregressive continuation.
- `analyze_generation.py`: checks generated files for silence/noise.
- `preview_audio.py`: exports WAV previews via `audio_play.py`.
- `inspect_npz.py`: prints dataset structure and metadata.

