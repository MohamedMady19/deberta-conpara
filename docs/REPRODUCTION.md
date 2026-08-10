# Reproduction

Hardware used: a single NVIDIA RTX 4090 (24 GB). Everything below runs on one
consumer GPU; nothing requires a cluster.

---

## 0 · Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. `sentencepiece` is required by the DeBERTa-v3 tokenizer.

---

## 1 · Obtain the corpora

See [`DATA.md`](DATA.md) for sources and licences. Expected layout:

```
data/raw/
├── hc3_plus/          train.csv, val_hc3_QA.csv, val_hc3_si.csv,
│                      test_hc3_QA.csv, test_hc3_si.csv
├── m4/                {arxiv,peerread,reddit,wikihow,wikipedia}_{generator}.jsonl
├── mage/              train.csv, validation.csv, test.csv
├── m4gt/              SubtaskA.jsonl, subtaskC_train_dev.jsonl
└── wikihow_generated/ (optional — see step 2)
```

RAID is fetched through the `raid-bench` package, not stored locally.

---

## 2 · (Optional) regenerate the WikiHow data

Only needed to reproduce the WikiHow coverage experiment. Requires API keys.

```bash
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... COHERE_API_KEY=... XAI_API_KEY=...

for g in gpt4o claude cohere grok; do
  python3 scripts/generate_wikihow.py --generator $g --n 2000
done
# local models need a GPU but no key
for g in llama8b mistral7b qwen14b; do
  python3 scripts/generate_wikihow.py --generator $g --n 2000
done
```

Approximate API cost: **€15–20** total for the four hosted generators. Expect
~4 h wall-clock for the hosted models and ~10 h for the three local ones on a
single 4090.

> One caution learned the hard way: models that bill internal reasoning tokens
> can consume a large output budget while returning very little text. Set a
> billing alert before starting, and check output length on a handful of
> samples before launching a full run.

---

## 3 · Extract features and build splits

```bash
python3 training/build_dataset.py --stage extract   # ~2.5 h
python3 training/build_dataset.py --stage split --plan-only
python3 training/build_dataset.py --stage split --export
```

`--plan-only` prints the full composition — per-source counts, class balance,
train/val sizes — without writing anything. Check it before committing to the
extraction.

Deterministic with seed 42.

---

## 4 · Train

```bash
python3 training/train_conpara.py --dry_run    # verify composition, then:
python3 training/train_conpara.py
```

Defaults: DeBERTa-v3-large, batch 4 × 8 accumulation (32 effective), lr 1e-5,
5 epochs, patience 3, seed 42. Roughly **2 days** at ~8 it/s on one 4090 for a
1 M-sample training set.

Feature scaling (`RobustScaler`) and mutual-information selection are refit on
the training split; the fitted parameters are stored in the checkpoint so that
inference reproduces them exactly.

**Select on your deployment metric.** The default saves on validation balanced
accuracy, which is *not* aligned with a TPR@1% objective — see
[`LIMITATIONS.md`](LIMITATIONS.md) §7. Pass `--select tpr1` if strict-FPR
performance is what you care about, and keep per-epoch snapshots so the choice
can be revisited.

---

## 5 · Evaluate

```bash
python3 evaluation/eval_cross_dataset.py    --checkpoint <ckpt>
python3 evaluation/eval_threshold_sweep.py  --checkpoint <ckpt>
python3 evaluation/raid_submission.py       --checkpoint <ckpt>
```

RAID submission runs inference over 672,000 test items — about **5 h** on one
4090. It writes `predictions.json` for the leaderboard.

---

## 6 · Regenerate figures

```bash
python3 scripts/make_figures.py
```

Writes vector PDF (`pdf.fonttype=42`, no rasterisation), editable SVG
(`svg.fonttype='none'`) and PNG into `figures/`.

---

## Migration checklist

Files to copy in from the working tree, with any cleanup needed:

| Destination | Source | Cleanup |
|---|---|---|
| `src/unicode_preprocessing_v2.py` | `~/Text/src/` | ✅ ready — includes the U+0440 fix |
| `src/features.py` | `FeatureExtractor` from `train_conpara_v24.py` | extract into its own module |
| `training/train_conpara.py` | `~/Text/train_conpara_v217.py` | strip absolute paths; add `--select` flag |
| `training/build_dataset.py` | `extract_v217_sources.py` + `build_v217_splits_v2.py` | merge behind `--stage` |
| `evaluation/eval_cross_dataset.py` | `~/Text/eval_ood_v216.py` | parameterise checkpoint |
| `evaluation/eval_threshold_sweep.py` | threshold-sweep script | parameterise checkpoint |
| `evaluation/raid_submission.py` | `raid_submission_v217.py` | parameterise checkpoint and τ |
| `scripts/generate_wikihow.py` | `~/Text/scripts/generate_wikihow_ai.py` | **remove hard-coded API keys** |
| `results/*.json` | RAID `results.json` files | keep per-attack/domain/generator breakdowns |

Before the first commit:

- [ ] no API keys anywhere (`git grep -iE "sk-|gsk_|xai-|AIza"`)
- [ ] no absolute paths (`git grep "/home/"`)
- [ ] no `.pt`, `.npz`, `.parquet` staged (covered by `.gitignore`)
- [ ] version numbers reconciled — the repo should tell one coherent story,
      not expose the full v1→v2.17 development history
