# Data

We do **not** redistribute any training corpus. Each is obtained from its
original source under its own licence; `training/build_dataset.py`
reconstructs the exact splits deterministically from what you download.

This is a deliberate choice. The corpora carry incompatible redistribution
terms, and shipping a merged dump would violate several of them. Reconstruction
from source is also more useful: it is verifiable, and it stays correct if an
upstream corpus is revised.

---

## Corpora

| Corpus | Obtain from | Licence | Notes |
|---|---|---|---|
| HC3 Plus | HuggingFace | see dataset card | official train/val/test splits used unchanged |
| M4 | M4 authors' GitHub release | see repository | English domains only: arXiv, PeerRead, Reddit, WikiHow, Wikipedia |
| MAGE | HuggingFace | see dataset card | official train/validation/test used unchanged |
| RAID | `pip install raid-bench` | see RAID repository | train split only; hidden test accessed via leaderboard |
| M4GT-Bench | M4GT-Bench GitHub | see repository | Subtask C train split used for PeerRead; Subtask A held out for evaluation |

### Splits

Where a corpus provides official splits (HC3 Plus, MAGE) we use them unchanged.
Where it does not, we construct a stratified 90/10 split:

| Corpus | Stratified by |
|---|---|
| M4 | domain × generator |
| RAID | domain × generator × attack |
| M4GT-Bench PeerRead | generator |
| WikiHow generations | generator |

**RAID splits are grouped, not stratified by row.** Each RAID article appears
with twelve variants (clean + eleven attacks) sharing an `adv_source_id`. All
variants of an article are assigned to the same side of the split; splitting by
row would place attack variants of the same underlying text in both train and
validation, inflating validation performance.

---

## WikiHow generations (ours)

To measure whether near-chance WikiHow detection reflected an intrinsic limit
or a coverage gap, we generated 13,958 WikiHow-style articles across seven
contemporary generators.

| Generator | n |
|---|---|
| Claude (Haiku) | 2,000 |
| GPT-4o | 2,000 |
| Grok | 2,000 |
| Cohere (command-a) | 2,000 |
| Llama-3.1-8B-Instruct (local) | 1,973 |
| Mistral-7B-Instruct (local) | 1,994 |
| Qwen-2.5-14B-Instruct (local) | 1,991 |

**Prompt.** Identical to the M4 template, so generations are directly
comparable with the existing M4 WikiHow data:

```
Please, generate wikihow article with length above 1000 characters
from title '{TITLE}' and headline '{STEPS}'
```

Titles and step outlines were extracted from the `prompt` field of the M4
WikiHow files, deduplicated by title.

**Quality filters** (all applied before acceptance):

- ≥ 100 words and ≥ 500 characters
- ≥ 3 sentences or explicit line structure
- rejected on refusal or meta-commentary patterns
  (`I cannot…`, `As an AI…`, `Note: this…`)
- truncated to 600 words at a sentence boundary
- sensitive-topic titles excluded (weapons, explosives, self-harm, and similar)

Temperature 0.7 and max 900 output tokens throughout. Every record retains its
generator id, prompt, temperature and timestamp.

**Release status.** These are our own artifact, but they are model outputs and
several providers place conditions on redistributing them. We are reviewing
each provider's terms; until that is resolved the generation script is released
and the outputs are not. Running `scripts/generate_wikihow.py` with your own
API keys reproduces them.

**One generator was excluded.** Gemini produced ~60-word responses regardless
of instruction or token budget: the model spends its budget on internal
reasoning tokens, which are billed but not returned. Every sample failed the
length filter, so the generator was dropped.

---

## A correction worth recording

An early version of this work assumed RAID contained ~146 K unique human
articles, based on distinct generation strings. It does not. RAID has **13,371
unique human articles**; the 160,452 human rows are twelve attack variants of
each. Any analysis treating variant count as coverage will overstate the
diversity of the human side by roughly an order of magnitude.
