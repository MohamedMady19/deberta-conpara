"""
DeBERTa-ConPara — public demo.

A research artifact. The output is supporting evidence for a human
judgement, never a verdict on its own.

Run locally:   python app.py
On Spaces:     this file is the entry point.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import gradio as gr
from transformers import DebertaV2Tokenizer, DebertaV2Model
from sklearn.preprocessing import RobustScaler
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from unicode_preprocessing_v2 import unicode_normalize
from features import FeatureExtractor

# --------------------------------------------------------------- config ---
REPO_ID   = os.environ.get("CONPARA_REPO", "mohamedmady/deberta-conpara")
CKPT_FILE = os.environ.get("CONPARA_CKPT", "model.pt")
BACKBONE  = "microsoft/deberta-v3-large"
HIDDEN    = 1024
MAX_LEN   = 512
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Below this many words the signal is close to insufficient — we say so
# rather than returning a confident-looking number.
MIN_RELIABLE_WORDS = 75
MIN_WORDS          = 25


# ---------------------------------------------------------------- model ---
class ConParaDetector(nn.Module):
    def __init__(self, nf=30):
        super().__init__()
        self.encoder = DebertaV2Model.from_pretrained(BACKBONE)
        hs = HIDDEN
        self.feat_proj = nn.Sequential(
            nn.Linear(nf, hs), nn.GELU(), nn.Dropout(0.1))
        self.feat_gate = nn.Sequential(
            nn.Linear(hs * 2, hs), nn.Tanh(), nn.Linear(hs, 1), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(hs * 2, hs // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hs // 2, 2))

    def forward(self, input_ids, attention_mask, features):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = enc.last_hidden_state[:, 0, :]
        f_ = self.feat_proj(features)
        gate = self.feat_gate(torch.cat([cls, f_], dim=1))
        return self.classifier(torch.cat([cls, gate * f_], dim=1))


print("loading checkpoint ...")
_ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=CKPT_FILE)
_ck = torch.load(_ckpt_path, map_location="cpu", weights_only=False)

MI_IDX = _ck["mi_selected_indices"]
SCALER = RobustScaler()
SCALER.center_ = _ck["scaler_center"]
SCALER.scale_ = _ck["scaler_scale"]
# v2.16 predates threshold-in-checkpoint; 0.920 is its calibrated tau*
THRESHOLD = float(_ck.get("threshold",
                          os.environ.get("CONPARA_TAU", 0.920)))

MODEL = ConParaDetector(nf=len(MI_IDX)).to(DEVICE)
MODEL.load_state_dict(_ck["model_state_dict"], strict=False)
MODEL.eval()

TOKENIZER = DebertaV2Tokenizer.from_pretrained(BACKBONE)
EXTRACTOR = FeatureExtractor()
print(f"ready — threshold {THRESHOLD:.2f}, {len(MI_IDX)} features")


# ------------------------------------------------------------ inference ---
def score(text: str) -> float:
    """Return P(AI-generated) for one text."""
    clean = unicode_normalize(text)
    raw = EXTRACTOR.extract_batch([clean])
    feats = SCALER.transform(np.nan_to_num(raw, nan=0.0,
                                           posinf=0.0, neginf=0.0))[:, MI_IDX]
    enc = TOKENIZER(clean, max_length=MAX_LEN, truncation=True,
                    padding="max_length", return_tensors="pt")
    with torch.no_grad():
        logits = MODEL(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE),
            torch.tensor(feats, dtype=torch.float32).to(DEVICE))
    return float(torch.softmax(logits.float(), dim=1)[0, 1])


def band(p: float) -> tuple[str, str, str]:
    """Map a probability to a coarse band. Deliberately coarse: the
    underlying scores are poorly calibrated at the extremes."""
    if p >= 0.90:
        return ("Strong signal of AI generation", "#B3261E", "strong-ai")
    if p >= THRESHOLD:
        return ("Some signal of AI generation", "#B26B00", "weak-ai")
    if p >= 0.20:
        return ("Inconclusive", "#5F5E5A", "unclear")
    return ("Consistent with human writing", "#1D6E56", "human")


GAUGE = """
<div class="cp-result {cls}">
  <div class="cp-verdict">{label}</div>
  <div class="cp-bar"><div class="cp-fill" style="width:{pct:.1f}%;
       background:{colour}"></div>
       <div class="cp-thresh" style="left:{tpct:.1f}%"></div></div>
  <div class="cp-scale"><span>consistent with human</span>
       <span>signal of AI</span></div>
  <div class="cp-meta">model score <b>{p:.3f}</b> ·
       decision threshold {t:.2f} · {words} words{trunc}</div>
</div>
"""

CAVEAT_SHORT = """
<div class="cp-note cp-warn">
  <b>Short text.</b> This passage is {words} words. In our evaluation, 93%
  of errors on the hardest benchmark occurred below 75 words — the result
  above should be treated as weak at best.
</div>
"""

CAVEAT_ALWAYS = """
<div class="cp-note">
  <b>This is supporting evidence, not a verdict.</b> The final judgement is
  a human one. In our own measurements this system produces false positives
  on 13–67% of formal academic prose depending on domain, so a positive
  score is not evidence of misconduct and must not be used as such.
</div>
"""


def analyse(text: str):
    if not text or not text.strip():
        return "", ""
    words = len(text.split())
    if words < MIN_WORDS:
        return (f'<div class="cp-note cp-warn">Please provide at least '
                f'{MIN_WORDS} words — {words} is too few for any meaningful '
                f'signal.</div>'), ""

    p = score(text)
    label, colour, cls = band(p)
    trunc = " · truncated to 512 tokens" if words > 380 else ""

    html = GAUGE.format(label=label, cls=cls, colour=colour,
                        pct=max(2.0, p * 100), tpct=THRESHOLD * 100,
                        p=p, t=THRESHOLD, words=words, trunc=trunc)
    notes = CAVEAT_ALWAYS
    if words < MIN_RELIABLE_WORDS:
        notes = CAVEAT_SHORT.format(words=words) + notes
    return html, notes


# ------------------------------------------------------------------ UI ----
CSS = """
.gradio-container {max-width: 900px !important; margin: auto;}
.cp-result {border:1px solid #E3E1DC; border-radius:10px; padding:18px 20px;
            margin-top:4px; background:#FDFDFC;}
.cp-verdict {font-size:17px; font-weight:600; margin-bottom:12px;}
.cp-result.strong-ai .cp-verdict {color:#B3261E;}
.cp-result.weak-ai   .cp-verdict {color:#B26B00;}
.cp-result.unclear   .cp-verdict {color:#5F5E5A;}
.cp-result.human     .cp-verdict {color:#1D6E56;}
.cp-bar {position:relative; height:10px; background:#EEEDE9;
         border-radius:5px; overflow:visible;}
.cp-fill {height:100%; border-radius:5px; transition:width .35s ease;}
.cp-thresh {position:absolute; top:-4px; width:2px; height:18px;
            background:#3A3936; opacity:.55;}
.cp-scale {display:flex; justify-content:space-between; font-size:11px;
           color:#8A8880; margin-top:6px;}
.cp-meta {font-size:12px; color:#5F5E5A; margin-top:12px;}
.cp-note {font-size:12.5px; line-height:1.55; color:#3A3936;
          background:#F6F5F2; border-left:3px solid #C9C6BF;
          padding:11px 14px; border-radius:0 6px 6px 0; margin-top:10px;}
.cp-note.cp-warn {background:#FFF6E8; border-left-color:#B26B00;}
footer {display:none !important;}
"""

INTRO = """
# DeBERTa-ConPara

Detection of AI-generated English text, robust to adversarial perturbation
and to shift across domains and generators.

Paste a passage below. **The result is one input to a human decision — it is
not proof of anything on its own.** Nothing you submit is stored or logged.
"""

EXAMPLES = [
    ["The mitochondrion is a double-membrane-bound organelle found in most "
     "eukaryotic organisms. Mitochondria generate most of the cell's supply "
     "of adenosine triphosphate, used as a source of chemical energy. They "
     "were first discovered by Albert von Kolliker in 1857, though their "
     "function remained unclear for several decades afterwards."],
    ["ok so I finally got round to fixing the sink and honestly what a "
     "nightmare. spent like two hours under there with a wrench and the "
     "thing still drips a bit. plumber wanted 180 quid which is mental for "
     "a twenty minute job. anyway it mostly works now, good enough."],
]

with gr.Blocks(title="DeBERTa-ConPara") as demo:
    gr.Markdown(INTRO)

    inp = gr.Textbox(lines=11, label="Text to analyse",
                     placeholder="Paste at least 75 words for a meaningful "
                                 "result...")
    with gr.Row():
        btn = gr.Button("Analyse", variant="primary", scale=2)
        clr = gr.Button("Clear", scale=1)

    out_html = gr.HTML()
    out_note = gr.HTML()

    gr.Examples(EXAMPLES, inputs=inp, label="Try an example")

    with gr.Accordion("How this works, and where it fails", open=False):
        gr.Markdown("""
**Pipeline.** Attack-aware Unicode normalisation (homoglyph, zero-width and
whitespace perturbations are neutralised before tokenisation) → 62 linguistic
and statistical features, reduced to 30 by mutual information → fused with
DeBERTa-v3-large representations through a learned gate.

**Benchmarks.** 98.9% TPR at 5% FPR on the RAID hidden test; 96.4% balanced
accuracy on M4GT-Bench; 99.6% / 86.4% / 94.6% on HC3-QA / HC3-SI / MAGE.

**Known failure modes.**

- *Short text.* 93% of errors on HC3-SI occur below 75 words. Below ~30 words
  the signal is close to information-theoretically insufficient.
- *Formal academic prose.* False positive rates of 13–67% depending on
  domain. Structured, impersonal writing reads as AI-like to this and to
  every detector we tested.
- *Strict operating points.* At 1% FPR performance drops sharply in some
  domains — product reviews worst, at 55.6% detection.
- *English only.* No claim is made about any other language.
- *Paraphrase.* Semantic rewriting is not defended against by preprocessing;
  robustness there comes from the model and is imperfect.

**Not a misconduct detector.** Detector output is not evidence of academic
dishonesty. Treat a positive score as weak circumstantial signal warranting
human review, never as a finding.

Code and full evaluation: github.com/mohamedmady/deberta-conpara
        """)

    btn.click(analyse, inputs=inp, outputs=[out_html, out_note])
    inp.submit(analyse, inputs=inp, outputs=[out_html, out_note])
    clr.click(lambda: ("", "", ""), outputs=[inp, out_html, out_note])

if __name__ == "__main__":
    demo.launch(css=CSS,
                theme=gr.themes.Soft(primary_hue="blue",
                                     neutral_hue="stone"))
