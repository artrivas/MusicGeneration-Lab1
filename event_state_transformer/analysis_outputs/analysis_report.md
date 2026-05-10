# Event-State Transformer Data Analysis

## Console summary

```text
Train sequences: 10,604; frames: 51,456,090; active-frame ratio: 0.2439
Unique non-empty chords: 344,137; total chord events: 12,548,532; entropy: 10.22 bits
Top-K coverage: K=5k 87.16%, K=10k 90.40%, K=20k 93.10%, K=50k 95.92%
Delta p95/p99/p99.5: 10.0/21.0/29.0; recommended max_delta: 30
Eval prefix/full sequences: 3/3; prefix matches full start: True
Metadata rows: 10604; matches train sequences: True
```

## NPZ keys and arrays

### train_npz

| key | shape | dtype | scalar |
|---|---:|---|---|
| `rolls_flat` | `[51456090, 88]` | `uint8` | `None` |
| `offsets` | `[10605]` | `int64` | `None` |
| `ids` | `[10604]` | `<U16` | `None` |
| `step_sec` | `[1]` | `float32` | `None` |
| `note_min` | `[1]` | `int16` | `None` |
| `note_max` | `[1]` | `int16` | `None` |
| `num_positions` | `[1]` | `int16` | `None` |
| `representation` | `[1]` | `<U16` | `None` |

### prefix_npz

| key | shape | dtype | scalar |
|---|---:|---|---|
| `rolls_flat` | `[300, 88]` | `uint8` | `None` |
| `offsets` | `[4]` | `int64` | `None` |
| `ids` | `[3]` | `<U32` | `None` |
| `step_sec` | `[1]` | `float32` | `None` |
| `note_min` | `[1]` | `int16` | `None` |
| `note_max` | `[1]` | `int16` | `None` |
| `num_positions` | `[1]` | `int16` | `None` |
| `representation` | `[1]` | `<U16` | `None` |
| `is_prefix` | `[1]` | `bool` | `None` |
| `prefix_steps` | `[1]` | `int32` | `None` |

### full_npz

| key | shape | dtype | scalar |
|---|---:|---|---|
| `rolls_flat` | `[12815, 88]` | `uint8` | `None` |
| `offsets` | `[4]` | `int64` | `None` |
| `ids` | `[3]` | `<U32` | `None` |
| `step_sec` | `[1]` | `float32` | `None` |
| `note_min` | `[1]` | `int16` | `None` |
| `note_max` | `[1]` | `int16` | `None` |
| `num_positions` | `[1]` | `int16` | `None` |
| `representation` | `[1]` | `<U16` | `None` |
| `is_prefix` | `[1]` | `bool` | `None` |
| `prefix_steps` | `[1]` | `int32` | `None` |

## A. Basic dataset information

- `rolls_flat`: shape `[51456090, 88]`, dtype `uint8`
- `offsets`: shape `[10605]`, dtype `int64`
- sequences: 10,604
- step_sec: 0.05000000074505806
- note range: 21 to 108
- num_positions: 88
- representation: onset
- total frames: 51,456,090
- active frames: 12,548,532
- empty frames: 38,907,558 (75.61%)
- average notes per frame: 0.4526
- average notes per active frame: 1.8557

## B. Sequence length statistics

| units | count | min | max | mean | median | p25 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frames | 10,604 | 402.000 | 11,977.000 | 4,852.517 | 4,406.000 | 2,920.000 | 6,297.250 | 8,871.400 | 10,182.000 | 11,541.910 |
| seconds | 10,604 | 20.100 | 598.850 | 242.626 | 220.300 | 146.000 | 314.863 | 443.570 | 509.100 | 577.096 |

| shorter than | sequence count | percentage |
|---:|---:|---:|
| 5s | 0 | 0.00% |
| 10s | 0 | 0.00% |
| 20s | 0 | 0.00% |
| 30s | 43 | 0.41% |
| 60s | 378 | 3.56% |

## C. Event-state statistics

- compression ratio active frames / total frames: 0.243869

| metric | count | min | max | mean | median | p90 | p95 | p99 | p99.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| active events / sequence | 10,604 | 52.000 | 5,741.000 | 1,183.377 | 945.000 | 2,355.000 | 2,906.000 | 3,902.790 | 4,277.000 |
| delta | 12,548,532 | 1.000 | 1,200.000 | 4.101 | 3.000 | 8.000 | 10.000 | 21.000 | 29.000 |

| delta threshold | count | percentage |
|---:|---:|---:|
| > 64 | 10,450 | 0.08% |
| > 128 | 1,901 | 0.02% |
| > 256 | 268 | 0.00% |
| > 512 | 22 | 0.00% |
| > 1200 | 0 | 0.00% |

- recommended max_delta from p99/p99.5: 30

## D. Chord vocabulary analysis

- unique non-empty chord states: 344,137
- total non-empty chord events: 12,548,532
- empirical chord entropy: 10.216 bits / 7.081 nats

| K | coverage | OOV | effective vocab size |
|---:|---:|---:|---:|
| 128 | 60.69% | 39.31% | 128 |
| 256 | 66.55% | 33.45% | 256 |
| 512 | 72.37% | 27.63% | 512 |
| 1,000 | 77.52% | 22.48% | 1,000 |
| 2,000 | 82.08% | 17.92% | 2,000 |
| 5,000 | 87.16% | 12.84% | 5,000 |
| 10,000 | 90.40% | 9.60% | 10,000 |
| 20,000 | 93.10% | 6.90% | 20,000 |
| 50,000 | 95.92% | 4.08% | 50,000 |

Top 20 chord frequencies:

| rank | MIDI pitches | count | percentage |
|---:|---|---:|---:|
| 1 | `[72]` | 237,722 | 1.89% |
| 2 | `[74]` | 230,473 | 1.84% |
| 3 | `[69]` | 229,952 | 1.83% |
| 4 | `[67]` | 225,595 | 1.80% |
| 5 | `[71]` | 202,473 | 1.61% |
| 6 | `[62]` | 201,010 | 1.60% |
| 7 | `[60]` | 199,424 | 1.59% |
| 8 | `[76]` | 196,466 | 1.57% |
| 9 | `[64]` | 195,175 | 1.56% |
| 10 | `[65]` | 185,677 | 1.48% |
| 11 | `[70]` | 183,405 | 1.46% |
| 12 | `[68]` | 173,030 | 1.38% |
| 13 | `[57]` | 172,734 | 1.38% |
| 14 | `[55]` | 170,731 | 1.36% |
| 15 | `[77]` | 168,181 | 1.34% |
| 16 | `[59]` | 165,789 | 1.32% |
| 17 | `[63]` | 164,628 | 1.31% |
| 18 | `[73]` | 164,444 | 1.31% |
| 19 | `[75]` | 164,085 | 1.31% |
| 20 | `[66]` | 162,134 | 1.29% |

## E. Note-level statistics

- notes per active frame: 1.000 | 17.000 | 1.856 | 1.000 | 4.000 | 4.000 | 6.000

Most common notes:

| rank | pitch_index | midi_pitch | count |
|---:|---:|---:|---:|
| 1 | 39 | 60 | 767,659 |
| 2 | 41 | 62 | 763,037 |
| 3 | 46 | 67 | 759,964 |
| 4 | 48 | 69 | 752,215 |
| 5 | 51 | 72 | 708,119 |
| 6 | 53 | 74 | 700,780 |
| 7 | 43 | 64 | 697,031 |
| 8 | 36 | 57 | 687,726 |
| 9 | 44 | 65 | 656,954 |
| 10 | 34 | 55 | 646,203 |
| 11 | 50 | 71 | 604,950 |
| 12 | 55 | 76 | 601,052 |
| 13 | 38 | 59 | 600,513 |
| 14 | 37 | 58 | 599,699 |
| 15 | 49 | 70 | 589,947 |
| 16 | 42 | 63 | 585,090 |
| 17 | 47 | 68 | 564,966 |
| 18 | 40 | 61 | 556,746 |
| 19 | 45 | 66 | 542,456 |
| 20 | 56 | 77 | 506,974 |

Least common notes:

| rank | pitch_index | midi_pitch | count |
|---:|---:|---:|---:|
| 1 | 87 | 108 | 12 |
| 2 | 86 | 107 | 31 |
| 3 | 85 | 106 | 55 |
| 4 | 0 | 21 | 129 |
| 5 | 84 | 105 | 378 |
| 6 | 2 | 23 | 742 |
| 7 | 1 | 22 | 1,465 |
| 8 | 83 | 104 | 1,823 |
| 9 | 82 | 103 | 6,397 |
| 10 | 81 | 102 | 7,097 |
| 11 | 3 | 24 | 7,883 |
| 12 | 4 | 25 | 8,459 |
| 13 | 6 | 27 | 13,254 |
| 14 | 80 | 101 | 15,721 |
| 15 | 7 | 28 | 16,430 |
| 16 | 5 | 26 | 16,996 |
| 17 | 79 | 100 | 21,458 |
| 18 | 9 | 30 | 26,327 |
| 19 | 78 | 99 | 28,390 |
| 20 | 8 | 29 | 30,175 |

Active-frame chord cardinality:

| notes in frame | percentage | count |
|---:|---:|---:|
| 1 | 54.97% | 6,898,357 |
| 2 | 23.38% | 2,934,180 |
| 3 | 10.77% | 1,351,040 |
| 4 | 6.00% | 752,586 |
| 5 | 2.86% | 358,263 |
| 6 | 1.28% | 161,224 |
| 7 | 0.48% | 59,828 |
| 8 | 0.18% | 23,135 |
| 9 | 0.05% | 6,824 |
| 10+ | 0.02% | 3,095 |

## F. Train/validation split diagnostic

- train sequences: 9,544
- validation sequences: 1,060
- train active-frame ratio: 0.243796
- validation active-frame ratio: 0.244538
- train unique chords: 324,181
- validation unique chords: 77,237
- train delta p95/p99: 10.000 / 21.000
- validation delta p95/p99: 10.000 / 21.000

## G. eval_set_01_prefix/full analysis

- prefix sequences: 3
- full sequences: 3
- same number of sequences: True
- prefix equals beginning of full: True
- prefix mismatch examples: []
- continuation length frames: 2,722.000 | 5,892.000 | 4,171.667 | 3,901.000 | 5,493.800 | 5,692.900 | 5,852.180
- continuation length seconds: 136.100 | 294.600 | 208.583 | 195.050 | 274.690 | 284.645 | 292.609
- prefix/full ratio: 0.017 | 0.035 | 0.026 | 0.025 | 0.033 | 0.034 | 0.035
- train active-frame ratio: 0.243869
- prefix active-frame ratio: 0.183333
- full active-frame ratio: 0.204604
- continuation active-frame ratio: 0.205114

Eval chord OOV under train vocabularies:

| K | prefix OOV | full OOV | prefix coverage | full coverage |
|---:|---:|---:|---:|---:|
| 5,000 | 7.27% | 3.43% | 92.73% | 96.57% |
| 10,000 | 3.64% | 2.59% | 96.36% | 97.41% |
| 20,000 | 1.82% | 1.72% | 98.18% | 98.28% |
| 50,000 | 0.00% | 0.72% | 100.00% | 99.28% |

## H. metadata.csv analysis

- pandas available: True
- rows: 10,604
- columns: `Id`, `num_steps`, `num_positions`
- matches train sequence count: True
- missing values: `{'Id': 0, 'num_steps': 0, 'num_positions': 0}`

Selected column summaries:

- `Id`: `{'unique': 10604, 'top_values': {'seq_000000': 1, 'seq_007063': 1, 'seq_007065': 1, 'seq_007066': 1, 'seq_007067': 1, 'seq_007068': 1, 'seq_007069': 1, 'seq_007070': 1, 'seq_007071': 1, 'seq_007072': 1, 'seq_007073': 1, 'seq_007074': 1, 'seq_007075': 1, 'seq_007076': 1, 'seq_007077': 1, 'seq_007078': 1, 'seq_007079': 1, 'seq_007080': 1, 'seq_007081': 1, 'seq_007064': 1}}`
- `num_steps`: `{'count': 10604.0, 'mean': 4852.516974726518, 'std': 2598.947002395843, 'min': 402.0, '25%': 2920.0, '50%': 4406.0, '75%': 6297.25, '90%': 8871.400000000001, '95%': 10182.0, '99%': 11541.909999999998, 'max': 11977.0}`

## Plots

- `sequence_lengths_seconds.png`
- `delta_hist_clipped_p99.png`
- `topk_coverage.png`
- `note_frequency.png`
- `notes_per_active_frame.png`

## Actionable recommendations for event_state_transformer

1. top_k=50000 is more defensible because top_k=20000 covers only 93.10%, but the effective size is 50,000 of 344,137 unique chords.
2. A reasonable max_delta is near the p99.5 delta, about 30 frames; compare this to the current 1,200 cap.
3. The event-state representation is moderately compressed: 24.39% of frames remain as events.
4. Validation loss is probably dominated by chord_ce because the empirical chord entropy is high; delta_ce can also matter if many deltas hit or approach the cap.
5. The train/val split does not show a large first-order mismatch in length, active ratio, or event count.
6. eval_set_01_prefix/full look broadly similar to train by active-frame ratio; chord OOV rates below give the stricter check.
7. Loss weights should be adjusted only after checking component losses; if chord_ce dominates, smaller top_k or chord smoothing is preferable before simply downweighting it.
8. Increasing samples_per_sequence or using more random crops is useful if sequences contain many events and validation uses center crops that miss train-time variety.
9. A frame-level Transformer/GRU baseline is worth considering if chord-token OOV is high or chord_ce remains dominant, because multi-label note prediction can generalize across unseen chord combinations.
