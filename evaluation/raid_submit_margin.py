#!/usr/bin/env python3
"""RAID hidden-test predictions for a no-feature checkpoint, scored as the
float32 logit margin. See usage at the bottom of main()."""
import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

DEV = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True


class NoFeat(nn.Module):
    def __init__(self, bb, hs=1024):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(bb)
        self.classifier = nn.Sequential(nn.Linear(hs, hs // 2), nn.GELU(),
                                        nn.Dropout(0.1), nn.Linear(hs // 2, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--date', default=time.strftime('%Y-%m-%d'))
    ap.add_argument('--backbone', default='microsoft/deberta-v3-large')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--raw', action='store_true')
    ap.add_argument('--ref', default='')
    ap.add_argument('--meta-template', default='')
    a = ap.parse_args()

    sys.path.insert(0, os.path.expanduser('~/Text'))
    from src.unicode_preprocessing_v2 import unicode_normalize as PN
    norm = lambda s: re.sub(r'\s+', ' ', PN(str(s))).strip()
    assert norm('te\u200bst h\u0435llo') == 'test hello', 'normaliser is not the real one'

    from raid.utils import load_data
    df = load_data(split='test')
    print(f'  RAID test rows: {len(df):,}')
    if not a.limit:
        assert len(df) == 672000, f'expected 672,000 rows, got {len(df):,}'
    ids = df['id'].tolist()
    texts = df['generation'].fillna('').astype(str).tolist()
    if a.limit:
        ids, texts = ids[:a.limit], texts[:a.limit]
    if not a.raw:
        t0 = time.time(); texts = [norm(t) for t in texts]
        print(f'  normalised {len(texts):,} texts in {time.time()-t0:.0f}s')

    tok = AutoTokenizer.from_pretrained(a.backbone)
    model = NoFeat(a.backbone)
    sd = torch.load(os.path.expanduser(a.ckpt), map_location='cpu',
                    weights_only=False)['model_state_dict']
    if any('feat_proj' in k for k in sd):
        sys.exit('checkpoint has a feature branch -- this script is for no-feature models')
    model.load_state_dict(sd, strict=True)
    model.to(DEV).eval()
    print(f'  loaded {Path(a.ckpt).name} (strict)')

    enc = tok(texts, truncation=True, max_length=512)['input_ids']
    order = np.argsort([len(e) for e in enc])
    margins = np.empty(len(enc), dtype=np.float32)
    pad = tok.pad_token_id
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), a.batch):
            idx = order[b:b + a.batch]
            L = max(len(enc[i]) for i in idx)
            iids = torch.full((len(idx), L), pad, dtype=torch.long)
            am = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                iids[r, :len(enc[i])] = torch.tensor(enc[i]); am[r, :len(enc[i])] = 1
            iids, am = iids.to(DEV), am.to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                cls = model.encoder(input_ids=iids, attention_mask=am).last_hidden_state[:, 0]
            lg = model.classifier(cls.float())
            margins[idx] = (lg[:, 1] - lg[:, 0]).cpu().numpy()
            if (b // a.batch) % 500 == 0:
                done = b + len(idx); rate = done / max(1e-9, time.time() - t0)
                print(f'  {done:>8,}/{len(order):,}  {rate:6.0f} rows/s  '
                      f'eta {(len(order)-done)/max(rate,1e-9)/60:5.1f} min', flush=True)

    print(f'\n  margins: min {margins.min():+.3f}  median {np.median(margins):+.3f}  '
          f'max {margins.max():+.3f}  unique {len(np.unique(margins)):,} of {len(margins):,}')
    print(f'  fraction > 0: {100*(margins>0).mean():.1f}%')
    if a.ref and os.path.exists(a.ref):
        from scipy.stats import spearmanr
        ref = {x['id']: x['score'] for x in json.load(open(a.ref))}
        both = [(ref[i], m) for i, m in zip(ids, margins) if i in ref]
        print(f'  Spearman vs reference ({len(both):,} ids): '
              f'{spearmanr([x for x,_ in both],[y for _,y in both]).statistic:.4f}')

    out = Path(os.path.expanduser(a.out)); out.mkdir(parents=True, exist_ok=True)
    json.dump([{'id': i, 'score': float(m)} for i, m in zip(ids, margins)],
              open(out / 'predictions.json', 'w'))
    meta = {}
    if a.meta_template and os.path.exists(os.path.expanduser(a.meta_template)):
        meta = json.load(open(os.path.expanduser(a.meta_template)))
    meta.update(detector_name=a.name, date_released=a.date)
    json.dump(meta, open(out / 'metadata.json', 'w'), indent=2)
    print(f'\n  wrote {out}  ({len(ids):,} rows)'); print(json.dumps(meta, indent=2))
    if a.limit:
        print('\n  LIMITED RUN -- do not submit this; rerun without --limit')


if __name__ == '__main__':
    main()
