# Results (for report / presentation)

**Last updated:** from `results/evaluation_report.json` (all artifacts on `dataset/val`, 8210 images, CPU, TF 2.21).

**Regenerate:** `python scripts/evaluate_results.py --full-val`

---

## Application: drone-mounted early warning

Edge classification: **Normal** / **Possible smoke** / **Possible fire**. Demo alert when top class is smoke or fire and confidence ≥ **55%** (same as `Classify.js`). INT8 TFLite is the recommended onboard format; TF.js is for browser demo.

---

## 1. Top-1 accuracy (each artifact tested separately)

| Model artifact | Top-1 acc | Agreement with Keras predictions |
|----------------|-----------|----------------------------------|
| `fire_smoke.keras` | **97.70%** | — (reference) |
| `fire_smoke_float32.tflite` | **97.70%** | 100.00% |
| `fire_smoke_dynamic.tflite` | **97.78%** | 99.52% |
| `fire_smoke_int8.tflite` | **97.50%** | 98.78% |
| TF.js (`public/model/`) | ~**97.7%** (expected) | Same graph as Keras; not batch-tested in CLI |

---

## 2. Per-class metrics (Keras = float32 TFLite)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| normal | 98.1% | 98.0% | 98.1% |
| smoke | 98.0% | 96.6% | 97.3% |
| fire | 97.0% | 98.3% | 97.7% |

**INT8 TFLite:** normal 97.5% / 98.0% / 97.8%; smoke 97.6% / 96.9% / 97.2%; fire 97.4% / 97.6% / 97.5%.

**Dynamic TFLite:** normal 98.2% / 98.3% / 98.2%; smoke 98.2% / 96.6% / 97.4%; fire 97.0% / 98.4% / 97.7%.

Keras confusion matrix (rows=true, cols=pred: normal, smoke, fire):

```text
[[2803,   27,   30],
 [  28, 2406,   56],
 [  25,   23, 2812]]
```

---

## 3. Hazard alert (smoke/fire, confidence ≥ 55%)

| Model | Alert precision | Alert recall | False alarm (normal) | Miss (smoke+fire) |
|-------|-----------------|--------------|----------------------|-------------------|
| Keras | 99.12% | 98.64% | 1.64% | 1.36% |
| TFLite float32 | 99.12% | 98.64% | 1.64% | 1.36% |
| TFLite dynamic | 99.30% | 98.71% | 1.29% | 1.29% |
| TFLite int8 | 99.13% | 98.28% | 1.61% | 1.72% |

---

## 4. Response time (single 224×224 frame, CPU)

| Model | Mean (ms) | p95 (ms) | FPS | Size (MB) |
|-------|-----------|----------|-----|-----------|
| Keras | 65.5 | 79.2 | 15.3 | 9.21 |
| TFLite float32 | 5.5 | 6.0 | 183.5 | 8.48 |
| TFLite dynamic | 23.2 | 24.6 | 43.2 | 2.41 |
| **TFLite int8** | **2.6** | **2.9** | **385.0** | **2.60** |
| TF.js (browser) | — | — | demo ~450 ms interval + 3-frame streak | 8.57 |

*Inference only; UAV adds camera + radio delay.*

---

## 5. Test method

1. Data: `dataset/val/{normal,smoke,fire}`, preprocessing `x = x/127 - 1`, 224×224.  
2. Script: `python scripts/evaluate_results.py --full-val` → `results/evaluation_report.json` / `.md`.  
3. Optional: `python scripts/benchmark_tflite.py` for TFLite-only latency check.  
4. Web: manual check with `npm run start-dev`.  
5. Limitations: desktop CPU ≠ onboard SoC; no flight test; demo threshold not for real fire alarms.

---

## 6. Suggested report paragraph

On 8,210 validation images, **Keras** and **float32 TFLite** reached **97.70%** top-1 accuracy (smoke recall **96.6%**, fire recall **98.3%**). **INT8 TFLite** reached **97.50%** at **2.6 ms** per frame (~385 FPS on our CPU), with model size **2.6 MB**, suitable for a UAV companion computer. Under the demo alert rule (≥55% on smoke/fire), alert recall was **98.3–98.7%** across exports, with **1.3–1.7%** false alarms on normal images. The browser demo uses TF.js with a **450 ms** sampling interval and **3-frame** confirmation to limit flicker.

---

## Files

| File | Content |
|------|---------|
| `results/evaluation_report.json` | Machine-readable full results |
| `results/evaluation_report.md` | Auto-generated tables |
| `RESULTS.md` | This summary for the report |
