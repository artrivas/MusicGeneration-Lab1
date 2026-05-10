# Recommended Event-State Transformer Training

Run from inside `event_state_transformer`:

```bash
python train.py \
  --data-dir .. \
  --out-dir runs/event_state_v2_k20k_d64 \
  --top-k 20000 \
  --max-delta 64 \
  --context-len 1024 \
  --samples-per-sequence 8 \
  --epochs 60 \
  --batch-size 16 \
  --lr 2e-4 \
  --warmup-steps 300 \
  --d-model 512 \
  --n-layers 8 \
  --n-heads 8 \
  --d-ff 2048 \
  --dropout 0.1 \
  --num-workers 4
```

If A100 memory allows, try `--batch-size 24` or `--batch-size 32`.

Recommended generation:

```bash
python generate.py \
  --data-dir .. \
  --checkpoint runs/event_state_v2_k20k_d64/best.pt \
  --out runs/event_state_v2_k20k_d64/eval_set_01_generated_event_state_v2.npz \
  --delta-temp 0.8 \
  --chord-temp 0.95 \
  --chord-top-p 0.92 \
  --note-temp 0.9 \
  --card-temp 0.9 \
  --max-notes 7 \
  --repeat-penalty 1.15 \
  --max-same-chord-run 3 \
  --prefer-note-head-prob 0.25
```

Smoke test:

```bash
python train.py \
  --data-dir .. \
  --out-dir runs/smoke_test \
  --top-k 1000 \
  --max-delta 64 \
  --context-len 256 \
  --samples-per-sequence 2 \
  --epochs 1 \
  --batch-size 2 \
  --max-train-steps 5
```

RoPE note: `--use-rope` is accepted and recorded, but this version keeps the existing absolute position embeddings. Replacing `nn.TransformerEncoderLayer` with rotary attention is postponed to avoid destabilizing this quick training pass.
