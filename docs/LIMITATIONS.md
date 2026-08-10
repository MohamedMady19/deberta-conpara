# Limitations

Stated plainly, because a detector's failure modes matter more than its
headline number.

---

## 1 · English only

Every experiment is English. The training corpora are English-primary, and we
make no claim about other languages. Whether Latin-script languages share
enough stylometric structure to permit transfer — and whether tonal or
non-alphabetic languages need fundamentally different approaches — is open.

## 2 · Strict false-positive rates

At 1% FPR performance is materially worse than at 5%, and unevenly so.

| RAID domain | TPR@5% | TPR@1% | gap |
|---|---|---|---|
| recipes | 98.9% | 98.9% | 0.0 pp |
| news | 99.0% | 99.0% | 0.0 pp |
| abstracts | 99.4% | 96.7% | 2.7 pp |
| wiki | 99.3% | 95.9% | 3.4 pp |
| books | 99.4% | 88.9% | 10.5 pp |
| reddit | 97.7% | 82.4% | 15.3 pp |
| poetry | 98.3% | 80.2% | 18.1 pp |
| **reviews** | **99.5%** | **55.6%** | **43.9 pp** |

The mechanism is a human false-positive tail, not a detection failure. At 1%
FPR the threshold is set by the most AI-like human texts in a domain; where
those score anomalously high the threshold is forced up and genuine detections
fall below it. Product reviews are short, formulaic and superlative-heavy —
close to AI register — and RAID provides only ~948 unique human review
articles, the smallest of any domain.

## 3 · Attacks that survive normalisation

Deterministic normalisation neutralises surface-form attacks completely, but
does nothing for semantic ones.

| Attack class | Neutralised | TPR@5%−TPR@1% gap |
|---|---|---|
| whitespace, zero-width, paragraph insertion | fully | ~7–8 pp |
| number, alternative spelling, misspelling | partially | ~8–9 pp |
| homoglyph, case perturbation | after fix ¹ | 13–27 pp |
| synonym substitution, paraphrase | not at all | 12–21 pp |

¹ See §6.

Paraphrase is close to a rewrite: mean word-level similarity 0.450,
character-level 0.113, ~172 word substitutions per document. Robustness here
comes from the model, not from preprocessing.

## 4 · Short text

93% of HC3-SI errors occur on texts under 75 words. Above that threshold the
error rate is ~0.8%. Below ~30 words the signal is close to
information-theoretically insufficient, and we do not consider this solvable by
better training data.

## 5 · Generator coverage

Cohere is the weakest generator family in every version we trained (96.2%
TPR@5%, 73.1% TPR@1%), and the gap has not closed. Detection quality tracks
generator representation in training, so any generator family absent from the
training mixture should be assumed weaker until measured.

## 6 · A preprocessing bug affecting published numbers

The homoglyph mapping contained an error. Cyrillic **р** (U+0440) was mapped to
Latin `r` — its *transliteration* — rather than `p`, which is what it visually
resembles. RAID's homoglyph attack substitutes `p → р`, so normalisation
"corrected" the attack into a different corruption:

```
development  →  develорment  →  develorment
play         →  рlay         →  rlay
```

This affected every result prior to the fix. It explains why homoglyph showed
the worst TPR@1% (72.6%) and worst AUROC (94.6%) of any attack while appearing
resolved at 5% FPR. Recovery of attacked text went from 0% to 100% once
corrected; four further missing lookalikes (U+0423, U+0455, U+0458, U+04BB)
were added at the same time.

Two consequences we want to be explicit about. First, the claim that
normalisation *fully* neutralises homoglyph attacks was not true of the
published system — the ablation gain was real, but the mechanism was partial.
Second, the map is now visual rather than transliterative: genuine Cyrillic
text will render as visual gibberish. That is the correct target for an
English-only detector whose map exists to defeat homoglyph attack, but it is a
deliberate trade-off.

## 7 · Checkpoint selection

Validation balanced accuracy at a fixed threshold is a poor selection criterion
for a TPR@1% objective. In one run the epoch with the best validation balanced
accuracy was substantially worse at 1% FPR than an earlier epoch — the model
had become overconfident, compressing scores toward the extremes, which
improves ranking marginally while worsening the human tail that strict-FPR
performance depends on. Selection should use the deployment metric directly.

## 8 · Not a misconduct detector

Our own measurements record 13–67% false positive rates on formal academic
prose depending on domain. Detector output is not evidence of academic
dishonesty and must not be used as such. Any deployment affecting individuals
should treat a positive score as weak circumstantial signal warranting human
review, never as a finding.
