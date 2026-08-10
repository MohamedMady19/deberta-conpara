#!/usr/bin/env python3
"""
Publication-quality figures for DeBERTa-ConPara.

Outputs true vector PDF (fonttype 42, no rasterisation) and editable SVG
(svg.fonttype='none') into figures/.

    python3 scripts/make_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------- style ----
plt.rcParams.update({
    'pdf.fonttype': 42,          # TrueType — text stays selectable
    'ps.fonttype': 42,
    'svg.fonttype': 'none',      # text as text, editable in Illustrator
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'figure.dpi': 150,
})

C = {
    'ours':    '#1D4E89',
    'ours_lt': '#5B8FC7',
    'rival':   '#C1666B',
    'neutral': '#8A8A8A',
    'accent':  '#D4913D',
    'good':    '#2E7D5B',
    'grid':    '#DDDDDD',
}

FIG = Path(__file__).resolve().parent.parent / 'figures'
FIG.mkdir(exist_ok=True)


def save(fig, name):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(FIG / f'{name}.{ext}', bbox_inches='tight',
                    dpi=300 if ext == 'png' else None)
    plt.close(fig)
    print(f'  figures/{name}.{{pdf,svg,png}}')


# ============================================================ FIGURE 1 =====
# The core claim: RAID specialists collapse cross-dataset; we do not.
def fig_cross_dataset():
    systems = ['Top academic\nRAID system', 'ADAL', 'TMR',
               'DeBERTa-ConPara\n(ours)']
    raid    = [99.8, 96.3, 95.8, 97.79]
    cross   = [53.9, 54.1, 72.3, 92.31]
    cols    = [C['rival'], C['rival'], C['rival'], C['ours']]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    x = np.arange(len(systems))
    w = 0.36

    b1 = ax.bar(x - w/2, raid,  w, label='RAID TPR@5% FPR',
                color=cols, edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + w/2, cross, w, label='HC3+MAGE balanced acc.',
                color=cols, edgecolor='white', linewidth=0.5, alpha=0.45,
                hatch='///')

    for bars, vals in ((b1, raid), (b2, cross)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 1.2, f'{v:.1f}',
                    ha='center', va='bottom', fontsize=7.5)

    ax.axhline(50, color=C['neutral'], lw=0.6, ls=':', zorder=0)
    ax.text(len(systems) - 0.48, 51.5, 'chance', fontsize=7,
            color=C['neutral'], ha='right')

    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel('Score (%)')
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis='y', color=C['grid'], lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower center', ncol=2,
              bbox_to_anchor=(0.5, 1.005))
    ax.set_title('Adversarial benchmark performance does not imply '
                 'cross-dataset robustness', pad=26, loc='left', fontsize=9.5)
    save(fig, 'fig1_cross_dataset_collapse')


# ============================================================ FIGURE 2 =====
# Preprocessing ablation + threshold insensitivity, side by side.
def fig_ablation_threshold():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7),
                                 gridspec_kw={'width_ratios': [1, 1.15]})

    # -- (a) preprocessing ablation --------------------------------------
    labels = ['Full model', 'w/o attack-aware\npreprocessing']
    vals   = [97.79, 84.50]
    bars = a1.bar(labels, vals, width=0.55,
                  color=[C['ours'], C['rival']],
                  edgecolor='white', linewidth=0.5)
    for b, v in zip(bars, vals):
        a1.text(b.get_x() + b.get_width()/2, v + 1, f'{v:.2f}',
                ha='center', fontsize=8)
    a1.annotate('', xy=(0.72, 84.5), xytext=(0.72, 97.79),
                arrowprops=dict(arrowstyle='<->', color=C['accent'], lw=1.1))
    a1.text(0.66, 91, '\u221213.29 pp', color=C['accent'], fontsize=8,
            va='center', ha='right')
    a1.set_ylabel('RAID TPR@5% FPR (%)')
    a1.set_ylim(0, 108)
    a1.grid(axis='y', color=C['grid'], lw=0.5)
    a1.set_axisbelow(True)
    a1.set_title('(a) Preprocessing ablation', loc='left', fontsize=9)

    # -- (b) fixed vs oracle threshold ------------------------------------
    ds     = ['HC3-QA', 'HC3-SI', 'MAGE']
    fixed  = [99.55, 83.76, 93.27]
    oracle = [99.63, 84.92, 93.43]
    y = np.arange(len(ds))
    h = 0.34
    a2.barh(y + h/2, fixed,  h, label='fixed $\\tau$',
            color=C['ours'], edgecolor='white', linewidth=0.5)
    a2.barh(y - h/2, oracle, h, label='oracle-optimal $\\tau$',
            color=C['ours_lt'], edgecolor='white', linewidth=0.5)
    for i, (f, o) in enumerate(zip(fixed, oracle)):
        a2.text(f + 0.6, i + h/2, f'{f:.2f}', va='center', fontsize=7.5)
        a2.text(o + 0.6, i - h/2, f'{o:.2f}', va='center', fontsize=7.5)
    a2.set_yticks(y)
    a2.set_yticklabels(ds)
    a2.set_xlim(75, 106)
    a2.set_ylim(-0.75, 2.9)
    a2.set_xlabel('Balanced accuracy (%)')
    a2.grid(axis='x', color=C['grid'], lw=0.5)
    a2.set_axisbelow(True)
    a2.legend(frameon=False, loc='lower center', ncol=2,
              bbox_to_anchor=(0.5, -0.02), handlelength=1.2)
    a2.set_title('(b) Threshold sensitivity  (mean gap +0.47 pp)',
                 loc='left', fontsize=9)

    fig.tight_layout(w_pad=2.0)
    save(fig, 'fig2_ablation_threshold')


# ============================================================ FIGURE 3 =====
# Training-data coverage drives per-domain detection (M4GT-Bench).
def fig_domain_coverage():
    domains = ['WikiHow', 'PeerRead', 'arXiv', 'Wikipedia', 'Reddit']
    v214    = [54.5, 66.9, 96.5, 96.1, 98.2]
    v216    = [95.6, 98.6, 95.7, 96.0, 97.7]

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    x = np.arange(len(domains))
    w = 0.36

    ax.bar(x - w/2, v214, w, label='before targeted curation',
           color=C['neutral'], edgecolor='white', linewidth=0.5, alpha=0.65)
    ax.bar(x + w/2, v216, w, label='after targeted curation',
           color=C['good'], edgecolor='white', linewidth=0.5)

    for i, (a, b) in enumerate(zip(v214, v216)):
        ax.text(i - w/2, a + 1.2, f'{a:.1f}', ha='center', fontsize=7.5)
        ax.text(i + w/2, b + 1.2, f'{b:.1f}', ha='center', fontsize=7.5)
        d = b - a
        if d > 5:
            ax.text(i, 108, f'+{d:.1f} pp', ha='center', fontsize=8,
                    color=C['good'], fontweight='bold')

    ax.axhline(50, color=C['neutral'], lw=0.6, ls=':', zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel('Balanced accuracy (%)')
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis='y', color=C['grid'], lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower center', ncol=2,
              bbox_to_anchor=(0.5, 1.005))
    ax.set_title('Per-domain detection tracks training-data coverage, '
                 'not model capacity', loc='left', pad=26, fontsize=9.5)
    save(fig, 'fig3_domain_coverage')


# ============================================================ FIGURE 4 =====
# Where the operating point hurts: TPR@5% vs TPR@1% per attack.
def fig_operating_point():
    attacks = ['homoglyph', 'synonym', 'upper_lower', 'paraphrase',
               'article_del.', 'alt_spelling', 'perp_misspell',
               'number', 'insert_para.', 'whitespace', 'zero_width']
    gap = [27.0, 21.0, 13.6, 12.3, 9.8, 8.9, 8.8, 8.0, 7.2, 7.8, 7.8]
    # attacks fully neutralised by deterministic normalisation
    neutralised = [False, False, False, False, False, False, False,
                   False, True, True, True]

    order = np.argsort(gap)[::-1]
    attacks = [attacks[i] for i in order]
    gap     = [gap[i] for i in order]
    neutralised = [neutralised[i] for i in order]

    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    cols = [C['good'] if n else C['rival'] for n in neutralised]
    y = np.arange(len(attacks))
    ax.barh(y, gap, 0.62, color=cols, edgecolor='white', linewidth=0.5)
    for i, g in enumerate(gap):
        ax.text(g + 0.4, i, f'{g:.1f}', va='center', fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(attacks)
    ax.invert_yaxis()
    ax.set_xlabel('TPR@5% $-$ TPR@1%  (pp)')
    ax.set_xlim(0, 31)
    ax.grid(axis='x', color=C['grid'], lw=0.5)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C['good'], label='neutralised by normalisation'),
        Patch(facecolor=C['rival'], label='survives normalisation')],
        frameon=False, loc='lower right')
    ax.set_title('Attacks that survive preprocessing dominate the '
                 'strict-FPR gap', loc='left', pad=8, fontsize=9.5)
    save(fig, 'fig4_operating_point_gap')


# ============================================================ FIGURE 5 =====
# RAID leaderboard trajectory across submitted versions.
def fig_version_trajectory():
    vers  = ['v2.6', 'v2.13', 'v2.14b', 'v2.16']
    tpr5  = [97.85, 97.78, 98.90, 98.88]
    auroc = [95.91, 97.15, 95.70, 96.91]

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    x = np.arange(len(vers))

    ax.plot(x, tpr5, 'o-', color=C['ours'], lw=1.4, ms=5,
            label='TPR@5% FPR')
    ax.plot(x, auroc, 's--', color=C['accent'], lw=1.4, ms=4.5,
            label='AUROC')

    for i, (t, a) in enumerate(zip(tpr5, auroc)):
        ax.text(i, t + 0.28, f'{t:.2f}', ha='center', fontsize=7.5,
                color=C['ours'])
        ax.text(i, a - 0.45, f'{a:.2f}', ha='center', fontsize=7.5,
                color=C['accent'])

    ax.set_xticks(x)
    ax.set_xticklabels(vers)
    ax.set_ylabel('Score (%)')
    ax.set_ylim(94.5, 100)
    ax.grid(axis='y', color=C['grid'], lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower right')
    ax.set_title('RAID hidden-test results across submitted versions',
                 loc='left', pad=8, fontsize=9.5)
    save(fig, 'fig5_version_trajectory')


if __name__ == '__main__':
    print('Generating figures...')
    fig_cross_dataset()
    fig_ablation_threshold()
    fig_domain_coverage()
    fig_operating_point()
    fig_version_trajectory()
    print(f'\nDone -> {FIG}')
