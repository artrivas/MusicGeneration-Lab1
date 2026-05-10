# MIREX-Like Token Continuation Experiment

This folder is a separate experimental approach for symbolic piano continuation. It does not replace the existing training or generation scripts in the project root.

The design is inspired by the MIREX 2025 symbolic piano continuation/RWKV direction:

- Use compact symbolic tokenization.
- Train with simple next-token prediction.
- Avoid predicting 88 independent binary outputs.
- Avoid huge observed-chord vocabularies.
- Generalize to unseen note combinations with `NOTE_ON` tokens.
- Decode back to `[T, 88]` so the teacher's `.npz` and `audio_play.py` pipeline still works.

The tokenizer is adapted to this lab's binary onset piano-roll format. Each event frame is represented as sorted `NOTE_ON_p` tokens followed by `CHORD_END`. Silent spans become one or more `TIME_SHIFT_n` tokens. The vocabulary is roughly `88 + max_shift + specials`, usually around 150 to 200 tokens.

This is different from a chord-vocabulary approach because it does not need to memorize every observed note combination. It is also different from the older 88-independent BCE approach because the model is a token-level language model trained with `CrossEntropyLoss`.

## Files

- `tokenizer_mirexlike.py`: `NOTE_ON` + `TIME_SHIFT` + `CHORD_END` tokenizer.
- `prepare_data.py`: converts `train.npz` into flattened token arrays.
- `dataset.py`: random sequence-level token window dataset.
- `model_transformer.py`: compact causal decoder-only Transformer.
- `model_rwkv.py`: clear RWKV placeholder.
- `train.py`: token-level CrossEntropy training.
- `generate.py`: continuation-specific inference from any prefix `.npz`.
- `analyze_generation.py`: roll statistics and optional real-continuation comparison.
- `preview_audio.py`: calls the project-root `audio_play.py`.
- `utils_npz.py`: reusable `.npz` helpers.

## Prepare

```bash
python newapproach/prepare_data.py \
  --train_npz train.npz \
  --cache_dir newapproach/cache \
  --max_shift 64
```

## Train

```bash
python newapproach/train.py \
  --cache_dir newapproach/cache \
  --out_dir newapproach/checkpoints \
  --model_type transformer \
  --context_len 1024 \
  --batch_size 8 \
  --epochs 20 \
  --steps_per_epoch 1000 \
  --lr 3e-4 \
  --d_model 384 \
  --n_layers 8 \
  --n_heads 8 \
  --dropout 0.15 \
  --weight_decay 0.01 \
  --label_smoothing 0.02 \
  --scheduler cosine \
  --warmup_steps 500 \
  --grad_clip 1.0 \
  --amp
```

## Generate

```bash
python newapproach/generate.py \
  --checkpoint newapproach/checkpoints/best.pt \
  --tokenizer newapproach/cache/tokenizer.json \
  --prefix_npz eval_set_01_prefix.npz \
  --out_npz newapproach/outputs/generated_less_repetitive.npz \
  --continuation_steps 2048 \
  --context_len 1024 \
  --temperature 0.95 \
  --top_k 80 \
  --top_p 0.97 \
  --density_control \
  --target_density_auto \
  --density_margin 0.03 \
  --density_strength 0.5 \
  --soft_max_silent_frames 64 \
  --hard_max_silent_frames 128 \
  --pattern_repetition_penalty 1.25 \
  --recent_pattern_window 64 \
  --max_same_pattern_repeats 6 \
  --note_frequency_penalty 0.15 \
  --recent_note_window 128 \
  --no_repeat_ngram_size 8 \
  --ngram_window 256 \
  --debug_generation_stats
```

Generation does not hard-code `eval_set_01`, the number of sequences, or the prefix length. It preserves the prefix exactly with:

```python
final_roll[:prefix_len] = prefix_roll
```

Local testing can match a full reference file length:

```bash
python newapproach/generate.py \
  --checkpoint newapproach/checkpoints/best.pt \
  --tokenizer newapproach/cache/tokenizer.json \
  --prefix_npz eval_set_01_prefix.npz \
  --target_full_npz eval_set_01_full.npz \
  --match_full_lengths \
  --out_npz newapproach/outputs/generated.npz
```

Do not use `--target_full_npz` during real teacher evaluation; it is only for local diagnostics.

## Analyze

```bash
python newapproach/analyze_generation.py \
  --npz newapproach/outputs/generated.npz \
  --prefix_npz eval_set_01_prefix.npz
```

With a full reference:

```bash
python newapproach/analyze_generation.py \
  --npz newapproach/outputs/generated.npz \
  --prefix_npz eval_set_01_prefix.npz \
  --full_npz eval_set_01_full.npz
```

## Preview

```bash
python newapproach/preview_audio.py \
  --npz newapproach/outputs/generated.npz \
  --index 0 \
  --out_wav newapproach/outputs/generated_0.wav \
  --num_steps 2048
```

## RWKV Status

`--model_type rwkv` is wired into the CLI, but `model_rwkv.py` intentionally raises:

```text
RWKV backend not implemented yet. Use --model_type transformer.
```

The Transformer backend is the working baseline.
