#!/usr/bin/env python3
"""
eval_cells_external.py -- evaluate every factorial cell on HC3-QA, HC3-SI,
MAGE and M4 under the paper's fixed-threshold protocol, so Table 4 can be
completed for all eight configurations rather than one.

For each cell it reports balanced accuracy at a threshold calibrated once on
the source validation split, AUROC, and TPR@1% FPR. It also reports the
oracle-threshold balanced accuracy, so the cost of fixing the threshold in
advance is visible per cell.

The inference-time preprocessing of each cell is applied to the benchmark
text as well, which is the point: a cell is defined by its inference
pipeline, not only by its weights.

Usage:
  python eval_cells_external.py --limit 25000
  python eval_cells_external.py --only N-rn          # re-run one cell
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MAXLEN = 512
T = Path(os.path.expanduser('~/Text'))
OLD = Path(os.path.expanduser('~/disk/from492/Text'))

# tag : (checkpoint dir, epoch glob, has features, normalise at inference)
BACKBONE = {'B-rn': 'bert-large-cased',
            'R-rn': 'roberta-large'}     # everything else: --backbone

CELLS = {
 'F-nn': ('models/conpara_v218',              'epoch5_*_s42.pt', 1, 1),
 'F-nr': ('models/conpara_v218',              'epoch5_*_s42.pt', 1, 0),
 'F-rr': ('models/conpara_v218b',             'epoch*_s42.pt',   1, 0),
 'F-rn': ('models/conpara_v218b',             'epoch*_s42.pt',   1, 1),
 'N-nn': ('models/conpara_v218_nofeat',       'epoch2_*_s42.pt', 0, 1),
 'N-nr': ('models/conpara_v218_nofeat',       'epoch2_*_s42.pt', 0, 0),
 'N-rr': ('models/conpara_v218_nofeat_raw',   'epoch2_*_s42.pt', 0, 0),
 'N-rn': ('models/conpara_v218_nofeat_raw',   'epoch2_*_s42.pt', 0, 1),
 'FR-rn': ('models/fullraid_deberta', 'epoch2_20260831_230118_s42.pt', 0, 1),
 'B-rn': ('models/bb_bert_raw',              'epoch2_*_s42.pt', 0, 1),
 'R-rn': ('models/bb_roberta_raw',           'epoch4_*_s42.pt', 0, 1),
}


class NoFeat(nn.Module):
    def __init__(self, bb, hs=1024):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(bb)
        self.classifier = nn.Sequential(nn.Linear(hs, hs // 2), nn.GELU(),
                                        nn.Dropout(0.1), nn.Linear(hs // 2, 2))
    def forward(self, i, m, f=None):
        return self.classifier(self.encoder(input_ids=i,
                                            attention_mask=m).last_hidden_state[:, 0])


class WithFeat(nn.Module):
    def __init__(self, bb, nf=30, hs=1024):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(bb)
        self.feat_proj = nn.Sequential(nn.Linear(nf, hs), nn.GELU(), nn.Dropout(0.1))
        self.feat_gate = nn.Sequential(nn.Linear(hs * 2, hs), nn.Tanh(),
                                       nn.Linear(hs, 1), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(hs * 2, hs // 2), nn.GELU(),
                                        nn.Dropout(0.1), nn.Linear(hs // 2, 2))
    def forward(self, i, m, f):
        cls = self.encoder(input_ids=i, attention_mask=m).last_hidden_state[:, 0]
        f_ = self.feat_proj(f)
        g = self.feat_gate(torch.cat([cls, f_], 1))
        return self.classifier(torch.cat([cls, g * f_], 1))


class DS(Dataset):
    def __init__(self, texts, feats, tok):
        self.t, self.f, self.tok = texts, feats, tok
    def __len__(self):
        return len(self.t)
    def __getitem__(self, i):
        e = self.tok(self.t[i] or ' ', max_length=MAXLEN, truncation=True,
                     padding='max_length', return_tensors='pt')
        f = torch.tensor(self.f[i], dtype=torch.float32) if self.f is not None \
            else torch.zeros(1)
        return e['input_ids'][0], e['attention_mask'][0], f


@torch.no_grad()
def score(model, tok, texts, feats, batch):
    dl = DataLoader(DS(texts, feats, tok), batch_size=batch, shuffle=False,
                    num_workers=4, pin_memory=True)
    out = []
    for i, m, f in dl:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = model(i.to(DEV), m.to(DEV), f.to(DEV))
        out.append((lg[:, 1] - lg[:, 0]).float().cpu().numpy())
    return np.concatenate(out)


def best_tau(m, y):
    grid = np.unique(np.quantile(m, np.linspace(0.001, 0.999, 1500)))
    b = [balanced_accuracy_score(y, (m > t).astype(int)) for t in grid]
    j = int(np.argmax(b))
    return float(grid[j]), 100 * float(b[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='microsoft/deberta-v3-large')
    ap.add_argument('--limit', type=int, default=25000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--only', default='')
    a = ap.parse_args()

    sys.path.insert(0, str(T))
    from src.unicode_preprocessing_v2 import unicode_normalize as PN
    import re as _re
    norm = lambda s: _re.sub(r'\s+', ' ', PN(str(s))).strip()
    raw = lambda s: str(s)

    # ---- benchmarks -------------------------------------------------
    B = {}
    hc3 = T / 'data/raw/hc3_plus'
    for nm, fn in (('HC3-QA', 'test_hc3_QA.csv'), ('HC3-SI', 'test_hc3_si.csv')):
        p = hc3 / fn
        if p.exists():
            df = pd.read_csv(p)
            low = {c.lower(): c for c in df.columns}
            B[nm] = ([str(x) for x in df[low['text']]],
                     df[low['is_ai']].astype(int).values)
    v = np.load(T / 'features/v218_raw/v218_val_62features.npz', allow_pickle=True)
    src = v['sources'].astype(str); yv = v['labels'].astype(int)
    for nm, keys in (('MAGE', ['MAGE']), ('M4', ['M4', 'm4_human_filler'])):
        k = np.isin(src, keys)
        B[nm] = ([str(t) for t in v['texts'][k]], yv[k])
    for nm in B:
        t_, y_ = B[nm]
        if len(t_) > a.limit:
            r = np.random.RandomState(0).choice(len(t_), a.limit, replace=False)
            B[nm] = ([t_[i] for i in r], np.asarray(y_)[r])
        print(f'  {nm:<8} {len(B[nm][0]):>7,} rows')

    tags = [a.only] if a.only else list(CELLS)
    res = {}
    for tag in tags:
        d, glob_, hasf, do_norm = CELLS[tag]
        cks = sorted((T / d).glob(glob_)) or sorted((OLD / d).glob(glob_))
        if not cks:
            print(f'  {tag}: NO CHECKPOINT at {T/d}/{glob_}')
            raise SystemExit(2)
        ck = torch.load(cks[-1], map_location='cpu', weights_only=False)
        sd = ck['model_state_dict']
        bb = BACKBONE.get(tag, a.backbone)
        tok = AutoTokenizer.from_pretrained(bb)
        model = (WithFeat(bb) if hasf else NoFeat(bb)).to(DEV)
        model.load_state_dict(sd, strict=True); model.eval()
        prep = norm if do_norm else raw
        print(f'\n=== {tag}  ({cks[-1].name}, {"norm" if do_norm else "raw"} inference)')

        # features, if this cell uses them
        def feats_for(texts):
            if not hasf:
                return None
            from sklearn.preprocessing import RobustScaler
            sc = RobustScaler(); sc.center_ = ck['scaler_center']; sc.scale_ = ck['scaler_scale']
            raise SystemExit(
                f'\n  {tag} needs the 62 handcrafted features recomputed for each\n'
                f'  benchmark. Run the existing feature extractor over HC3/MAGE\n'
                f'  first, or evaluate the four no-feature cells only with\n'
                f'  --only N-nn (etc.).')

        # calibrate tau* on the source validation split
        rs = np.random.RandomState(42)
        keep = np.concatenate([rs.choice(np.where(yv == c)[0],
                                         min(12000, (yv == c).sum()), replace=False)
                               for c in (0, 1)])
        mv = score(model, tok, [prep(t) for t in v['texts'][keep]],
                   feats_for(None), a.batch)
        tau, bav = best_tau(mv, yv[keep])
        print(f'  tau* {tau:+.3f}   source-val BA {bav:.2f}')

        res[tag] = {'tau': tau, 'val_ba': bav, 'bench': {}}
        for nm, (txt, yy) in B.items():
            m = score(model, tok, [prep(t) for t in txt], None, a.batch)
            yy = np.asarray(yy)
            ba = 100 * balanced_accuracy_score(yy, (m > tau).astype(int))
            _, bo = best_tau(m, yy)
            au = roc_auc_score(yy, m)
            hu, ai = m[yy == 0], m[yy == 1]
            t1 = 100 * (ai > np.quantile(hu, 0.99)).mean() if len(hu) > 20 else float('nan')
            res[tag]['bench'][nm] = {'ba': ba, 'ba_oracle': bo,
                                     'auroc': float(au), 'tpr1': float(t1)}
            print(f'    {nm:<8} BA {ba:6.2f}  (oracle {bo:6.2f})  '
                  f'AUROC {au:.4f}  TPR@1% {t1:6.2f}')
        del model; torch.cuda.empty_cache()

    out = Path(os.path.expanduser('~/queue_out/cells_external.json'))
    out.parent.mkdir(exist_ok=True)
    old = {}
    if out.exists():
        try: old = json.load(open(out))
        except Exception: pass
    old.update(res)
    json.dump(old, open(out, 'w'), indent=2, default=float)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
