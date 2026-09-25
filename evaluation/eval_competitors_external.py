#!/usr/bin/env python3
"""
eval_competitors_external.py — open-checkpoint RAID leaderboard detectors under
the SAME fixed-threshold protocol used for our cells (eval_cells_external.py):

    1. score our source validation split with the competitor
    2. choose tau* ONCE on that split (same rule as our cells)
    3. hold tau* fixed on HC3-QA, HC3-SI, MAGE, M4 (and anything else in --sets)
    4. report balanced accuracy @tau*, AUROC, TPR@1/5% FPR, oracle-tau gap,
       native-threshold balanced accuracy, per-domain breakdown, bootstrap CIs

Models are discovered from the RAID leaderboard metadata (full HF ids, not the
truncated strings in the terminal listing), then filtered by --models.

Label direction for every model with a custom head was taken from its model
card (checked 2026-09-16). A polarity check on the validation split aborts
the run if AUROC < 0.5 (i.e. the AI index is wrong) unless --allow-flip.

Usage (one model per GPU; scores are cached, so the run can be resumed):
  conda activate v218
  python eval_competitors_external.py --spec sets.json --list
  CUDA_VISIBLE_DEVICES=0 python eval_competitors_external.py --spec sets.json \
      --models ADAL TMR --out ~/queue_out/competitors
  python eval_competitors_external.py --spec sets.json --out ~/queue_out/competitors --report

sets.json (fill in paths/columns to match what eval_cells_external.py loads):
{
  "val":  {"path": "~/Text/.../val.parquet", "text": "text", "label": "label",
           "ai_value": 1, "groups": ["source", "domain"]},
  "sets": {
    "hc3_qa": {"path": "...", "text": "text", "label": "label", "ai_value": 1, "groups": ["domain"]},
    "hc3_si": {...}, "mage": {...}, "m4": {...}
  },
  "normalizer": "preprocess_module:normalize_text"      # optional, our inference-time normaliser
}
"""
from __future__ import annotations

import argparse, glob, hashlib, importlib, json, os, re, sys, time
from pathlib import Path

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # as eval_cells_external.py


def _no_compile(cfg):
    """ModernBERT-family (MELD's Ettin backbone, GeorgeDrayson/modernbert) torch.compiles
    by default on CUDA; that forks inductor workers and is fragile on torch 2.2."""
    if getattr(cfg, "model_type", "") == "modernbert":
        cfg.reference_compile = False
    return cfg

# --------------------------------------------------------------------------
# Environment banner — the first line must show the v218 env (see handover:
# jobs launched without `conda activate v218` die silently).
# --------------------------------------------------------------------------
def banner():
    env = os.environ.get("CONDA_DEFAULT_ENV", "<none>")
    import torch, transformers
    print(f"[env] conda={env} python={sys.version.split()[0]} torch={torch.__version__} "
          f"transformers={transformers.__version__} cuda={torch.cuda.is_available()} "
          f"visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}", flush=True)
    if env != "v218":
        print("[env] WARNING: not running in conda env v218", flush=True)


# --------------------------------------------------------------------------
# Leaderboard discovery
# --------------------------------------------------------------------------
# Typos in leaderboard metadata (verified against the HF API)
HF_ID_FIX = {
    "MayZhou/e5-small-lora-ai-generated-detectord": "MayZhou/e5-small-lora-ai-generated-detector",
}
# Checkpoints whose training data overlaps one of our eval sets -> flag, not OOD
TRAIN_OVERLAP = {
    "GeorgeDrayson/modernbert-ai-detection-raid-mage": ["mage"],
}

HF_RE = re.compile(r"huggingface\.co/(?!spaces/)([\w.\-]+/[\w.\-]+)")

def discover(raid_dir: Path):
    """Return {detector_name: hf_id} for every submission with an HF *model* link."""
    out = {}
    for m in sorted(glob.glob(str(raid_dir / "*" / "metadata.json"))):
        meta = json.load(open(m))
        link = (meta.get("huggingface_link") or "").strip()
        mm = HF_RE.search(link)
        if not mm:
            continue
        raw_id = mm.group(1).rstrip("/")
        hf_id = HF_ID_FIX.get(raw_id, raw_id)
        out[meta.get("detector_name", Path(m).parent.name)] = {
            "hf_id": hf_id, "submission": Path(m).parent.name, "link": link,
            "id_corrected_from": raw_id if hf_id != raw_id else None}
    return out


# --------------------------------------------------------------------------
# Adapters.  Each returns a callable: list[str] -> np.ndarray of P(AI)-like scores
# (monotone in "more AI" is all that matters; tau* is calibrated per model).
# --------------------------------------------------------------------------
AI_LBL = re.compile(r"(^|[^a-z])(ai|fake|machine|generated|gpt|chatgpt|llm|synthetic)([^a-z]|$)", re.I)
HU_LBL = re.compile(r"(human|real)", re.I)

# Card-verified overrides (index of the AI class in the softmax)
AI_INDEX_OVERRIDE = {
    "Shushant/ADAL-detector-large": 0,     # card: probs[0] = P(AI)
    "Oxidane/tmr-ai-text-detector": 1,     # card: probs[1] = P(AI)
    "TrustSafeAI/RADAR-Vicuna-7B": 0,      # RADAR demo: log_softmax(...)[:, 0] = AI
    "MayZhou/e5-small-lora-ai-generated-detector": 1,   # card: LABEL_1 = AI
}
# Models whose score measures HUMAN-ness (card-verified) -> negate
HUMAN_SCORE = {
    "ShantanuT01/BERT-tiny-RAID",          # card: "outputs a score corresponding to the likelihood that the text is human"
}
MAX_LEN = {  # card-stated where available, else backbone limit
    "desklib/ai-text-detector-v1.01": 768,
    "anon-review-meld-2026/meld": None,    # read from meld_config.json
}


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _batched(texts, bs, fn):
    """Length-sorted batching; returns scores in original order."""
    order = np.argsort([len(t) for t in texts])[::-1]
    out = np.empty(len(texts), dtype=np.float64)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        out[idx] = fn([texts[j] for j in idx])
        if (i // bs) % max(1, 2000 // bs) == 0:
            print(f"    {min(i + bs, len(texts)):>7}/{len(texts)}", flush=True)
    return out


def adapter_hf_seqcls(hf_id, revision, bs):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from huggingface_hub import list_repo_files
    files = list_repo_files(hf_id, revision=revision)
    dev = _device()
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    if "adapter_config.json" in files:                       # LoRA adapter (e5-small-lora)
        from peft import AutoPeftModelForSequenceClassification
        model = AutoPeftModelForSequenceClassification.from_pretrained(hf_id, revision=revision)
        model = model.merge_and_unload()
    else:
        from transformers import AutoConfig
        cfg = _no_compile(AutoConfig.from_pretrained(hf_id, revision=revision))
        extra = {"attn_implementation": "eager"} if cfg.model_type == "modernbert" else {}
        model = AutoModelForSequenceClassification.from_pretrained(hf_id, revision=revision,
                                                                   config=cfg, **extra)
    model.to(dev).eval()
    n_lab = model.config.num_labels
    id2label = {int(k): v for k, v in (model.config.id2label or {}).items()}
    if hf_id in AI_INDEX_OVERRIDE:
        ai_idx, why = AI_INDEX_OVERRIDE[hf_id], "model-card override"
    elif n_lab == 1:
        ai_idx, why = None, "single logit"
        if hf_id in HUMAN_SCORE:
            why += " (HUMAN score per model card -> negated)"
    else:
        hits = [i for i, l in id2label.items() if AI_LBL.search(str(l)) and not HU_LBL.search(str(l))]
        if len(hits) == 1:
            ai_idx, why = hits[0], f"id2label {id2label}"
        else:
            ai_idx, why = 1, f"DEFAULT index 1 (id2label={id2label}) — verify via polarity check"
    print(f"  [{hf_id}] labels={n_lab} ai_index={ai_idx} ({why})", flush=True)
    max_len = min(getattr(tok, "model_max_length", 512) or 512, 512)
    max_len = MAX_LEN.get(hf_id) or max_len

    @torch.no_grad()
    def f(batch):
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(dev)
        enc.pop("token_type_ids", None) if "roberta" in model.config.model_type else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(**enc).logits.float()
        if ai_idx is None:
            sgn = -1.0 if hf_id in HUMAN_SCORE else 1.0
            return (sgn * logits[:, 0]).cpu().numpy()              # raw logit, AI-positive
        # log-odds of the AI class = our cells' (lg[:,1]-lg[:,0]) for 2 classes;
        # avoids ties from saturated softmax probabilities
        lo = torch.log_softmax(logits, -1)
        other = torch.cat([lo[:, :ai_idx], lo[:, ai_idx + 1:]], 1).logsumexp(1)
        return (lo[:, ai_idx] - other).cpu().numpy()
    stats = {"nan_fallback": 0}

    def safe(batch):
        out = np.asarray(f(batch), dtype=np.float64)
        bad = np.where(~np.isfinite(out))[0]
        if len(bad):                           # same torch-2.2 sdpa/padding NaN guard as MELD
            stats["nan_fallback"] += len(bad)
            cfg = model.config
            prev = cfg._attn_implementation
            cfg._attn_implementation = "eager"
            try:
                for i in bad:
                    out[i] = f([batch[i]])[0]
            finally:
                cfg._attn_implementation = prev
        return out

    return (lambda texts: _batched(texts, bs, safe)), {"ai_index": ai_idx, "ai_index_source": why,
                                                      "max_len": max_len, "score": "AI log-odds",
                                                      "native_tau": 0.0, "nan_stats": stats}


def adapter_desklib(hf_id, revision, bs):
    """Card code: mean-pooled backbone + Linear(H,1), sigmoid, max_len 768."""
    # Plain nn.Module + strict state-dict load: the card's PreTrainedModel subclass
    # breaks under transformers>=5 (missing all_tied_weights_keys).
    import torch, torch.nn as nn
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    d = snapshot_download(hf_id, revision=revision)

    class DesklibAIDetectionModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
        def forward(self, input_ids, attention_mask=None):
            h = self.model(input_ids, attention_mask=attention_mask)[0]
            m = attention_mask.unsqueeze(-1).expand(h.size()).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            return self.classifier(pooled)

    dev = _device()
    tok = AutoTokenizer.from_pretrained(d)
    model = DesklibAIDetectionModel(AutoConfig.from_pretrained(d))
    sd = {}
    for f in sorted(glob.glob(f"{d}/*.safetensors")):
        from safetensors.torch import load_file
        sd.update(load_file(f))
    if not sd:
        sd = torch.load(f"{d}/pytorch_model.bin", map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # position_ids buffers are harmless; anything else means a silent random init
    bad = [k for k in list(missing) + list(unexpected) if "position_ids" not in k]
    if bad:
        raise RuntimeError(f"desklib state-dict mismatch: {bad[:10]}")
    model.to(dev).eval()
    max_len = 768

    @torch.no_grad()
    def f(batch):
        # card pads to max_length; mean pooling is mask-weighted, so dynamic padding is equivalent
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            lg = model(enc["input_ids"], enc["attention_mask"])
        return lg.float().view(-1).cpu().numpy()                   # raw logit
    return (lambda texts: _batched(texts, bs, f)), {"ai_index": "single logit", "max_len": max_len,
                                                    "score": "logit", "native_tau": 0.0}


def adapter_meld(hf_id, revision, bs):
    """Card code (v5): Ettin-400m + style/prototype head; top-rho token mean; raw score s."""
    import torch, torch.nn as nn
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    d = snapshot_download(hf_id, revision=revision)

    class Meld(nn.Module):
        def __init__(self, model_dir):
            super().__init__()
            self.cfg = json.load(open(f"{model_dir}/meld_config.json"))
            r, H = self.cfg["style_rank"], self.cfg["backbone_hidden_size"]
            # eager, not the card's sdpa: torch-2.2 sdpa emits NaN on padded batches and the
            # single-text fallback was far too slow. Same maths, batched.
            self.backbone = AutoModel.from_config(_no_compile(AutoConfig.from_pretrained(model_dir)),
                                                  attn_implementation="eager")
            self.style_proj = nn.Linear(H, r, bias=False)
            self.style_ln = nn.LayerNorm(r)
            self.human_anchors = nn.Parameter(torch.zeros(self.cfg["n_human_anchors"], r))
            self.family_protos = nn.Parameter(torch.zeros(self.cfg["n_families"], r))
            self.family_bias = nn.Parameter(torch.zeros(self.cfg["n_families"]))
            self.log_tau = nn.Parameter(torch.zeros(()))
            self.op_protos = nn.Parameter(torch.zeros(self.cfg["n_ops"], r))
            self.op_bias = nn.Parameter(torch.zeros(self.cfg["n_ops"]))
            self.load_state_dict(load_file(f"{model_dir}/model.safetensors"), strict=True)
            self.eval()

        @torch.no_grad()
        def score(self, texts, tokenizer, device):
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=self.cfg["max_length"],
                            return_special_tokens_mask=True).to(device)
            valid = enc["attention_mask"].bool() & ~enc["special_tokens_mask"].bool()
            h = self.backbone(input_ids=enc["input_ids"],
                              attention_mask=enc["attention_mask"]).last_hidden_state.float()
            u = self.style_ln(self.style_proj(h))
            tau = self.log_tau.clamp(-4.0, 4.0).exp()
            def sqdist(u, p):
                return ((u * u).sum(-1, keepdim=True) - 2.0 * u @ p.t()
                        + (p * p).sum(-1).view(1, 1, -1))
            human = torch.logsumexp(-tau * sqdist(u, self.human_anchors), -1, keepdim=True)
            family = -tau * sqdist(u, self.family_protos) + self.family_bias.view(1, 1, -1)
            per_token = self.cfg["tau_agg"] * torch.logsumexp(
                (family - human).clamp(-30.0, 30.0) / self.cfg["tau_agg"], dim=-1)
            x = per_token.masked_fill(~valid, torch.finfo(per_token.dtype).min)
            x, _ = x.sort(dim=1, descending=True)
            k = (valid.sum(1).clamp(min=1) * self.cfg["rho"]).ceil().clamp(min=1).long()
            keep = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < k.unsqueeze(1)
            s = torch.where(keep, x, torch.zeros_like(x)).sum(1) / k.float()
            return s.cpu().numpy()                     # raw score; native threshold is on s

    dev = _device()
    model = Meld(d).to(dev)
    tok = AutoTokenizer.from_pretrained(d)
    native = model.cfg.get("score_offsets", {}).get("overall", {})
    stats = {"texts": 0, "nan_fallback": 0}

    def safe(batch):
        """Batched sdpa; any non-finite row is re-scored alone (no padding) with eager
        attention. torch-2.2 sdpa + padded ModernBERT batches can emit NaN."""
        s = np.asarray(model.score(batch, tok, dev), dtype=np.float64)
        stats["texts"] += len(batch)
        bad = np.where(~np.isfinite(s))[0]
        if len(bad):
            first = stats["nan_fallback"] == 0
            stats["nan_fallback"] += len(bad)
            cfg = model.backbone.config
            prev = cfg._attn_implementation
            cfg._attn_implementation = "eager"
            try:
                for i in bad:
                    s[i] = model.score([batch[i]], tok, dev)[0]
            finally:
                cfg._attn_implementation = prev
            if first:
                print(f"    [MELD] {len(bad)}/{len(batch)} NaN in batch -> eager single-text "
                      f"fallback; still NaN: {int((~np.isfinite(s[bad])).sum())}", flush=True)
        return s

    return (lambda texts: _batched(texts, max(1, bs // 4), safe)), \
        {"ai_index": "raw score s (sigmoid-monotone)", "max_len": model.cfg["max_length"],
         "native_thresholds_raw": native, "score": "raw s",
         "native_tau": native.get("fpr_0.01"), "min_words_note": "card: unreliable under 100 words", "nan_stats": stats}


def adapter_superannotate(hf_id, revision, bs):
    """Needs `pip install git+https://github.com/superannotateai/generated_text_detector`."""
    import torch
    from transformers import AutoTokenizer
    from generated_text_detector.utils.model.roberta_classifier import RobertaClassifier
    from generated_text_detector.utils.preprocessing import preprocessing_text
    dev = _device()
    model = RobertaClassifier.from_pretrained(hf_id, revision=revision).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)

    @torch.no_grad()
    def f(batch):
        batch = [preprocessing_text(t) for t in batch]
        enc = tok(batch, add_special_tokens=True, max_length=512, padding="longest",
                  truncation=True, return_token_type_ids=True, return_tensors="pt").to(dev)
        _, logits = model(**enc)
        return logits.float().squeeze(1).cpu().numpy()             # raw logit
    return (lambda texts: _batched(texts, bs, f)), {"ai_index": "single logit", "max_len": 512,
                                                    "score": "logit", "native_tau": 0.0,
                                                    "preprocessing": "vendor preprocessing_text"}


CUSTOM = {
    "desklib/ai-text-detector-v1.01": adapter_desklib,
    "anon-review-meld-2026/meld": adapter_meld,
    "SuperAnnotate/ai-detector": adapter_superannotate,
}


def build_scorer(hf_id, revision, bs):
    return CUSTOM.get(hf_id, adapter_hf_seqcls)(hf_id, revision, bs)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_set(cfg, max_n=None, seed=0):
    import pandas as pd
    p = os.path.expanduser(cfg["path"])
    if p.endswith(".npz"):
        # our v218 split: texts/labels + sources/domains/generators/attacks/decodings
        z = np.load(p, allow_pickle=True)
        keep_cols = [k for k in z.files if k != "features"]
        df = pd.DataFrame({k: z[k] for k in keep_cols})
        n0 = len(df)
        for col, vals in (cfg.get("include") or {}).items():
            df = df[df[col].astype(str).isin([str(v) for v in vals])]
        for col, vals in (cfg.get("exclude") or {}).items():
            df = df[~df[col].astype(str).isin([str(v) for v in vals])]
        print(f"  [npz] {Path(p).name}: {n0} -> {len(df)} rows "
              f"(include={cfg.get('include')}, exclude={cfg.get('exclude')})", flush=True)
        if len(df) == 0:
            raise ValueError(f"filter matched 0 rows in {p}; check source names")
    elif p.endswith(".parquet"):
        df = pd.read_parquet(p)
    elif p.endswith(".jsonl"):
        df = pd.read_json(p, lines=True)
    elif p.endswith(".csv"):
        df = pd.read_csv(p)
    else:
        raise ValueError(f"unsupported file type: {p}")
    low = {c.lower(): c for c in df.columns}              # case-insensitive, as eval_cells
    tcol, lcol = low[cfg["text"].lower()], low[cfg["label"].lower()]
    df = df.rename(columns={tcol: "_text"})
    cfg = {**cfg, "label": lcol}
    df["_text"] = [str(x) for x in df["_text"]]          # eval_cells: str(x), NaN -> 'nan'
    df["_y"] = (df[cfg["label"]].astype(int) == cfg.get("ai_value", 1)).astype(int)
    df = df.reset_index(drop=True)
    # eval_cells_external.py: B[nm] capped with RandomState(0).choice(n, limit)
    lim = cfg.get("limit")
    if lim and len(df) > lim:
        r = np.random.RandomState(cfg.get("limit_seed", 0)).choice(len(df), lim, replace=False)
        df = df.iloc[r].reset_index(drop=True)
    # eval_cells_external.py: tau* calibration sample, class 0 then class 1
    cs = cfg.get("calib_sample")
    if cs:
        y = df["_y"].values
        rs = np.random.RandomState(cs.get("seed", 42))
        keep = np.concatenate([rs.choice(np.where(y == c)[0], min(cs["per_class"], int((y == c).sum())),
                                         replace=False) for c in (0, 1)])
        df = df.iloc[keep].reset_index(drop=True)
        max_n = None
    if max_n and len(df) > max_n:            # stratified subsample (label x first group)
        keys = ["_y"] + cfg.get("groups", [])[:1]
        frac = max_n / len(df)
        df = (df.groupby(keys, group_keys=False)
                .apply(lambda g: g.sample(frac=frac, random_state=seed))
                .reset_index(drop=True))
    return df


def fingerprint(texts):
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8", "ignore")); h.update(b"\0")
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def best_tau_grid(s, y):
    """Verbatim port of eval_cells_external.best_tau (strict m > t)."""
    from sklearn.metrics import balanced_accuracy_score
    grid = np.unique(np.quantile(s, np.linspace(0.001, 0.999, 1500)))
    b = [balanced_accuracy_score(y, (s > t).astype(int)) for t in grid]
    j = int(np.argmax(b))
    return float(grid[j]), float(b[j])


def choose_tau(y, s, rule, target_fpr):
    from sklearn.metrics import roc_curve
    if rule == "grid":
        return best_tau_grid(s, y)[0]
    fpr, tpr, thr = roc_curve(y, s)
    if rule == "bal_acc":                    # max (TPR + TNR)/2  == max Youden J
        i = int(np.argmax(tpr - fpr))
    elif rule == "fpr":
        ok = np.where(fpr <= target_fpr)[0]
        i = int(ok[-1])
    else:
        raise ValueError(rule)
    return float(thr[i]) if np.isfinite(thr[i]) else float(np.max(s))


def bal_acc(y, s, tau):
    pred = s > tau                            # strict, as in eval_cells_external.py
    tpr = pred[y == 1].mean() if (y == 1).any() else np.nan
    tnr = (~pred[y == 0]).mean() if (y == 0).any() else np.nan
    if np.isnan(tpr) or np.isnan(tnr):
        return float(np.nanmax([tpr, tnr])), float(tpr), float(tnr), True
    return float((tpr + tnr) / 2), float(tpr), float(tnr), False


def metric_block(y, s, tau, native_tau=None, n_boot=1000, seed=0):
    from sklearn.metrics import roc_auc_score, roc_curve
    ba, tpr, tnr, single = bal_acc(y, s, tau)
    out = {"n": int(len(y)), "n_ai": int(y.sum()), "n_human": int((1 - y).sum()),
           "bal_acc": ba, "ai_recall": tpr, "human_recall": tnr, "single_class": single}
    if single:
        return out
    out["auroc"] = float(roc_auc_score(y, s))
    fpr, tp, _ = roc_curve(y, s)
    out["tpr@1fpr"] = float(np.interp(0.01, fpr, tp))
    out["tpr@5fpr"] = float(np.interp(0.05, fpr, tp))
    oracle = best_tau_grid(s, y)[1]           # same oracle definition as our cells
    hu, ai = s[y == 0], s[y == 1]             # same empirical TPR@1% as our cells
    out["tpr1_emp"] = float((ai > np.quantile(hu, 0.99)).mean()) if len(hu) > 20 else float("nan")
    out["oracle_bal_acc"] = oracle
    out["oracle_gap"] = oracle - ba
    if native_tau is not None:
        out["bal_acc_native"] = bal_acc(y, s, native_tau)[0]
    if n_boot <= 0:
        return out
    rng = np.random.default_rng(seed)             # stratified bootstrap of bal_acc @tau*
    ia, ih = np.where(y == 1)[0], np.where(y == 0)[0]
    pa, ph = (s[ia] >= tau), (s[ih] < tau)
    bs = [(pa[rng.integers(0, len(ia), len(ia))].mean() +
           ph[rng.integers(0, len(ih), len(ih))].mean()) / 2 for _ in range(n_boot)]
    out["bal_acc_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def breakdown(df, s, tau, groups):
    res = {}
    for g in groups:
        if g not in df.columns:
            continue
        res[g] = {}
        for k, idx in df.groupby(g).indices.items():
            y = df["_y"].values[idx]
            res[g][str(k)] = metric_block(y, s[idx], tau, n_boot=0) if len(idx) >= 20 else {"n": len(idx)}
    return res


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="sets.json (see docstring)")
    ap.add_argument("--raid-dir", default="~/raid-fork/leaderboard/submissions")
    ap.add_argument("--models", nargs="*", default=None,
                    help="detector names or HF ids (substring match); default: all discovered")
    ap.add_argument("--exclude", nargs="*", default=["RoBERTa-base (GPT2)", "RoBERTa-large (GPT2)",
                                                      "RoBERTa (ChatGPT)"],
                    help="legacy baselines skipped by default")
    ap.add_argument("--revision", nargs="*", default=[],
                    help="pin revisions: hf_id=sha ... (recorded in output either way)")
    ap.add_argument("--out", default="~/queue_out/competitors")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-max", type=int, default=0,
                    help="stratified val subsample (0 = full split, as for our cells)")
    ap.add_argument("--tau-rule", choices=["grid", "bal_acc", "fpr"], default="grid",
                    help="MUST match the rule used in eval_cells_external.py")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--normalize", choices=["off", "on", "both"], default="off",
                    help="apply our inference-time normaliser to competitor inputs")
    ap.add_argument("--allow-flip", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="load + score 64 val texts, then stop")
    ap.add_argument("--report", action="store_true", help="aggregate existing results only")
    a = ap.parse_args()

    spec = json.load(open(os.path.expanduser(a.spec)))
    out = Path(os.path.expanduser(a.out)); out.mkdir(parents=True, exist_ok=True)

    if a.report:
        return report(out)

    found = discover(Path(os.path.expanduser(a.raid_dir)))
    if a.list:
        for k, v in found.items():
            fix = f"  (fixed from {v['id_corrected_from']})" if v["id_corrected_from"] else ""
            print(f"{k:<40} {v['hf_id']:<55} [{v['submission']}]{fix}")
        print(f"\n{len(found)} submissions with an HF model link (spaces excluded).")
        return

    banner()
    sel = {k: v for k, v in found.items() if k not in a.exclude}
    if a.models:
        sel = {k: v for k, v in sel.items()
               if any(m.lower() in k.lower() or m.lower() in v["hf_id"].lower() for m in a.models)}
    if not sel:
        sys.exit("no models matched --models; run --list")
    print(f"[models] {len(sel)}: " + ", ".join(sel), flush=True)
    for k, v in sel.items():
        if TRAIN_OVERLAP.get(v["hf_id"]):
            print(f"[models] NOTE {k}: trained on {TRAIN_OVERLAP[v['hf_id']]} -> not OOD there", flush=True)
    revs = dict(r.split("=", 1) for r in a.revision)

    norm_fn = None
    if a.normalize != "off":
        mod, fn = spec["normalizer"].split(":")
        sys.path.insert(0, os.path.expanduser(spec.get("code_root", "~/Text")))
        _pn = getattr(importlib.import_module(mod), fn)
        norm_fn = lambda t: re.sub(r"\s+", " ", _pn(str(t))).strip()   # == eval_cells `norm`
    variants = {"off": ["raw"], "on": ["norm"], "both": ["raw", "norm"]}[a.normalize]

    print("[data] loading", flush=True)
    val = load_set(spec["val"], a.val_max or None)
    sets = {k: load_set(c) for k, c in spec["sets"].items()}
    if a.dry_run:
        val = val.sample(64, random_state=0).reset_index(drop=True)
        sets = {}
    for k, d in [("val", val)] + list(sets.items()):
        print(f"  {k:<8} n={len(d):>7} ai={d['_y'].sum():>7} fp={fingerprint(d['_text'])}", flush=True)
        if d["_y"].nunique() < 2 and k == "val":
            sys.exit("validation split must contain both classes")

    from huggingface_hub import HfApi
    api = HfApi()
    for name, info in sel.items():
        hf_id = info["hf_id"]
        slug = re.sub(r"[^\w.-]", "_", hf_id)
        try:
            sha = revs.get(hf_id) or api.model_info(hf_id).sha
        except Exception as e:
            print(f"[{name}] SKIP: cannot resolve {hf_id}: {e}", flush=True); continue
        print(f"\n=== {name}  {hf_id}@{sha[:10]} ===", flush=True)
        t0 = time.time()
        try:
            score, meta = build_scorer(hf_id, sha, a.batch_size)
        except Exception as e:
            print(f"[{name}] SKIP: load failed: {type(e).__name__}: {e}", flush=True); continue

        for var in variants:
            prep = (lambda ts: [norm_fn(t) for t in ts]) if var == "norm" else (lambda ts: list(ts))
            cache = out / "scores" / slug / var
            cache.mkdir(parents=True, exist_ok=True)

            def get(tag, df):
                fp = fingerprint(df["_text"])
                f = cache / f"{tag}_{fp}_{sha[:10]}.npy"
                if f.exists() and not a.dry_run:
                    return np.load(f)
                print(f"  scoring {tag} ({var}) n={len(df)}", flush=True)
                s = score(prep(df["_text"].tolist()))
                if not a.dry_run and np.isfinite(s).all():
                    np.save(f, s)
                return s

            yv = val["_y"].values
            sv = get("val", val)
            if not np.isfinite(sv).all():
                print(f"[{name}] SKIP: {int((~np.isfinite(sv)).sum())}/{len(sv)} non-finite val "
                      f"scores ({meta.get('nan_stats')})", flush=True)
                break
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(yv, sv)
            print(f"  [polarity] val AUROC={val_auc:.4f}", flush=True)
            if val_auc < 0.5:
                if not a.allow_flip:
                    print(f"[{name}] SKIP: val AUROC {val_auc:.3f} < 0.5 -> AI direction likely "
                          f"wrong ({meta}). Fix the adapter or rerun with --allow-flip.", flush=True)
                    break
                print("  [polarity] FLIPPING scores (--allow-flip)", flush=True)
                meta["flipped"] = True
            flip = meta.get("flipped", False)
            sv_ = -sv if flip else sv
            if a.dry_run:
                print(f"  dry-run OK: {meta}  scores[:5]={np.round(sv[:5], 4)}"); continue

            tau = choose_tau(yv, sv_, a.tau_rule, a.target_fpr)
            native = None if flip else meta.get("native_tau")   # logit 0 / MELD 1%-FPR cut
            res = {"detector": name, "hf_id": hf_id, "revision": sha, "variant": var,
                   "leaderboard_submission": info["submission"], "adapter": meta,
                   "id_corrected_from": info.get("id_corrected_from"),
                   "train_overlap_sets": TRAIN_OVERLAP.get(hf_id, []),
                   "protocol": {"tau_rule": a.tau_rule, "target_fpr": a.target_fpr,
                                "val_n": int(len(val)), "val_fp": fingerprint(val["_text"]),
                                "val_max": a.val_max},
                   "tau_star": tau, "native_tau": native,
                   "val": metric_block(yv, sv_, tau, native),
                   "val_breakdown": breakdown(val, sv_, tau, spec["val"].get("groups", [])),
                   "sets": {}}
            for k, df in sets.items():
                s = get(k, df); s = -s if flip else s
                if not np.isfinite(s).all():
                    print(f"[{name}] WARNING: {int((~np.isfinite(s)).sum())} non-finite scores on {k} "
                          f"-> set skipped", flush=True)
                    continue
                y = df["_y"].values
                res["sets"][k] = {"overall": metric_block(y, s, tau, native),
                                  "breakdown": breakdown(df, s, tau, spec["sets"][k].get("groups", []))}
                o = res["sets"][k]["overall"]
                print(f"  {k:<8} bal@tau*={o['bal_acc']*100:6.2f}  "
                      f"AUROC={o.get('auroc', float('nan'))*100:6.2f}  "
                      f"gap={o.get('oracle_gap', float('nan'))*100:5.2f}", flush=True)
            ext = [res["sets"][k]["overall"]["bal_acc"] for k in ("hc3_qa", "hc3_si", "mage")
                   if k in res["sets"]]
            if len(ext) == 3:
                res["avg_bal_acc_hc3qa_hc3si_mage"] = float(np.mean(ext))
                print(f"  AVG(HC3-QA, HC3-SI, MAGE) = {np.mean(ext)*100:.2f}", flush=True)
            res["wall_seconds"] = round(time.time() - t0, 1)
            # one file per (model, variant): parallel jobs never overwrite each other
            json.dump(res, open(out / f"{slug}__{var}.json", "w"), indent=1)

        del score
        import gc, torch; gc.collect(); torch.cuda.empty_cache()


def report(out: Path):
    rows = []
    for f in sorted(out.glob("*__*.json")):
        r = json.load(open(f))
        g = lambda k, m: r["sets"].get(k, {}).get("overall", {}).get(m, float("nan")) * 100
        rows.append((r.get("avg_bal_acc_hc3qa_hc3si_mage", float("nan")) * 100, r["detector"],
                     r["variant"], r["revision"][:10], r["tau_star"],
                     g("hc3_qa", "bal_acc"), g("hc3_si", "bal_acc"), g("mage", "bal_acc"),
                     g("m4", "bal_acc"), g("mage", "auroc"), g("m4", "auroc")))
    rows.sort(key=lambda x: -np.nan_to_num(x[0], nan=-1))
    hdr = f"{'avg3':>6} {'detector':<34} {'var':<4} {'rev':<10} {'tau*':>8} " \
          f"{'HC3QA':>6} {'HC3SI':>6} {'MAGE':>6} {'M4':>6} {'MAGEau':>6} {'M4au':>6}"
    lines = [hdr] + [f"{r[0]:6.2f} {r[1][:34]:<34} {r[2]:<4} {r[3]:<10} {r[4]:8.4f} "
                     + " ".join(f"{v:6.2f}" for v in r[5:]) for r in rows]
    print("\n".join(lines))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\n{len(rows)} result files -> {out/'summary.txt'}")


if __name__ == "__main__":
    main()
