# Evaluation report (all model artifacts)

**Device:** CPU; same val set and preprocessing (x/127-1) for all artifacts
**TensorFlow:** 2.21.0
**Val images:** 8210

## 1. Top-1 accuracy comparison

| Model artifact | Top-1 acc | vs Keras label agreement |
|----------------|-----------|---------------------------|
| fire_smoke.keras | 97.70% | 100.00% |
| fire_smoke_float32.tflite | 97.70% | 100.00% |
| fire_smoke_dynamic.tflite | 97.78% | 99.52% |
| fire_smoke_int8.tflite | 97.50% | 98.78% |

| TF.js (`public/model/`) | Not batch-tested in CLI | Same graph export as Keras; float32 TFLite matches Keras 100% on val |

*TF.js automated subset eval skipped (use browser demo). Treat accuracy as **~97.7%** (parity with Keras / float32 TFLite).*

## 2. Hazard alert summary (rule: smoke/fire, confidence ≥ 55%)

| Model | Alert precision | Alert recall | False alarm (normal) | Miss (smoke+fire) |
|-------|-----------------|--------------|----------------------|-------------------|
| fire_smoke.keras | 99.12% | 98.64% | 1.64% | 1.36% |
| fire_smoke_float32.tflite | 99.12% | 98.64% | 1.64% | 1.36% |
| fire_smoke_dynamic.tflite | 99.30% | 98.71% | 1.29% | 1.29% |
| fire_smoke_int8.tflite | 99.13% | 98.28% | 1.61% | 1.72% |

## 3. Per-model detail

### fire_smoke.keras

- Top-1: **97.70%**
- Alert recall: 98.64% | False alarm (normal): 1.64%
- Latency: 65.54 ms (p95 79.16 ms, 15.3 FPS)

Confusion matrix (rows=true, cols=pred: normal, smoke, fire):

```
[[2803   27   30]
 [  28 2406   56]
 [  25   23 2812]]
```

| Class | P | R | F1 |
|-------|---|---|-----|
| normal | 0.981 | 0.980 | 0.981 |
| smoke | 0.980 | 0.966 | 0.973 |
| fire | 0.970 | 0.983 | 0.977 |

### fire_smoke_float32.tflite

- Top-1: **97.70%**
- Alert recall: 98.64% | False alarm (normal): 1.64%
- Latency: 5.45 ms (p95 5.95 ms, 183.5 FPS)

Confusion matrix (rows=true, cols=pred: normal, smoke, fire):

```
[[2803   27   30]
 [  28 2406   56]
 [  25   23 2812]]
```

| Class | P | R | F1 |
|-------|---|---|-----|
| normal | 0.981 | 0.980 | 0.981 |
| smoke | 0.980 | 0.966 | 0.973 |
| fire | 0.970 | 0.983 | 0.977 |

### fire_smoke_dynamic.tflite

- Top-1: **97.78%**
- Alert recall: 98.71% | False alarm (normal): 1.29%
- Latency: 23.17 ms (p95 24.55 ms, 43.2 FPS)

Confusion matrix (rows=true, cols=pred: normal, smoke, fire):

```
[[2810   20   30]
 [  27 2405   58]
 [  24   23 2813]]
```

| Class | P | R | F1 |
|-------|---|---|-----|
| normal | 0.982 | 0.983 | 0.982 |
| smoke | 0.982 | 0.966 | 0.974 |
| fire | 0.970 | 0.984 | 0.977 |

### fire_smoke_int8.tflite

- Top-1: **97.50%**
- Alert recall: 98.28% | False alarm (normal): 1.61%
- Latency: 2.6 ms (p95 2.88 ms, 385.0 FPS)

Confusion matrix (rows=true, cols=pred: normal, smoke, fire):

```
[[2803   33   24]
 [  28 2412   50]
 [  43   27 2790]]
```

| Class | P | R | F1 |
|-------|---|---|-----|
| normal | 0.975 | 0.980 | 0.978 |
| smoke | 0.976 | 0.969 | 0.972 |
| fire | 0.974 | 0.976 | 0.975 |

## 4. Model size (MB)

```json
{
  "fire_smoke.keras": 9.21,
  "fire_smoke_float32.tflite": 8.48,
  "fire_smoke_dynamic.tflite": 2.41,
  "fire_smoke_int8.tflite": 2.6,
  "tfjs_public_model_total_mb": 8.57
}
```

## 5. Reproduce

```text
python scripts/evaluate_results.py --full-val
python scripts/benchmark_tflite.py
```