"""
SmolVLA Weight Visualizer
=========================
Visualize weights from: wuc1/bi_so101_flatten-and-fold-the-rag-32layers-smolvla_base-0403-model

Usage:
    pip install torch safetensors huggingface_hub matplotlib numpy
    python visualize_smolvla_weights.py

This will download the model (if not cached) and generate a comprehensive weight visualization.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch

# ── 1. Load model weights ──────────────────────────────────────────────────────
REPO_ID = "wuc1/bi_so101_flatten-the-rag-0326-base_on_0323-3_model-0328-2"

def load_weights():
    """Try safetensors first, then pytorch bin."""
    try:
        from safetensors import safe_open
        from huggingface_hub import snapshot_download
        path = snapshot_download(REPO_ID)
        st_files = glob.glob(os.path.join(path, "*.safetensors"))
        if st_files:
            weights = {}
            for f in st_files:
                with safe_open(f, framework="pt", device="cpu") as sf:
                    for k in sf.keys():
                        weights[k] = sf.get_tensor(k)
            print(f"Loaded {len(weights)} tensors from safetensors")
            return weights
        # fallback to bin
        bin_files = glob.glob(os.path.join(path, "*.bin"))
        if bin_files:
            weights = {}
            for f in bin_files:
                weights.update(torch.load(f, map_location="cpu"))
            print(f"Loaded {len(weights)} tensors from .bin")
            return weights
    except Exception as e:
        print(f"snapshot_download failed: {e}")

    # Try single file download
    from huggingface_hub import hf_hub_download
    try:
        f = hf_hub_download(REPO_ID, "model.safetensors")
        from safetensors import safe_open
        weights = {}
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                weights[k] = sf.get_tensor(k)
        print(f"Loaded {len(weights)} tensors")
        return weights
    except:
        pass
    f = hf_hub_download(REPO_ID, "pytorch_model.bin")
    weights = torch.load(f, map_location="cpu")
    print(f"Loaded {len(weights)} tensors")
    return weights

weights = load_weights()

# ── 2. Categorize layers ───────────────────────────────────────────────────────
vision_keys = [k for k in weights if any(x in k.lower() for x in ["vision", "image", "visual", "dino", "siglip", "clip"])]
language_keys = [k for k in weights if any(x in k.lower() for x in ["lm", "llm", "language", "text", "embed", "transformer.h", "model.layers"])]
action_keys = [k for k in weights if any(x in k.lower() for x in ["action", "flow", "expert", "decoder"])]
other_keys = [k for k in weights if k not in vision_keys + language_keys + action_keys]

# If categorization is poor, fallback by prefix
if len(vision_keys) == 0 and len(language_keys) == 0:
    print("Auto-categorizing by prefix...")
    for k in weights:
        parts = k.split(".")
        prefix = parts[0] if parts else k
        # will be grouped in "other" for now

def count_params(keys):
    return sum(weights[k].numel() for k in keys)

print(f"\n{'='*60}")
print(f"Model: {REPO_ID}")
print(f"Total parameters: {sum(v.numel() for v in weights.values()):,}")
print(f"Total tensors: {len(weights)}")
print(f"Vision layers: {len(vision_keys)} ({count_params(vision_keys):,} params)")
print(f"Language layers: {len(language_keys)} ({count_params(language_keys):,} params)")
print(f"Action layers: {len(action_keys)} ({count_params(action_keys):,} params)")
print(f"Other layers: {len(other_keys)} ({count_params(other_keys):,} params)")
print(f"{'='*60}\n")

# ── 3. Visualization ──────────────────────────────────────────────────────────

BG = '#0f0f1a'
ACCENT = '#00d2ff'

fig = plt.figure(figsize=(22, 28), facecolor=BG)
fig.suptitle(
    f'SmolVLA Weight Visualization\n{REPO_ID}',
    fontsize=18, color='white', fontweight='bold', y=0.99,
    fontfamily='monospace'
)

gs = GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.35,
              top=0.96, bottom=0.03, left=0.06, right=0.97)

def style_ax(ax, title):
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
    ax.set_facecolor(BG)
    ax.tick_params(colors='#888', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#333')

# ── 3a. Global weight histogram ───────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
all_w = torch.cat([v.float().flatten() for v in weights.values()]).numpy()
# Clip for display
clip = np.percentile(np.abs(all_w), 99.5)
ax.hist(all_w[np.abs(all_w) < clip], bins=200, color=ACCENT, alpha=0.85, edgecolor='none')
ax.axvline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.6)
ax.set_xlabel("Value", color='#aaa', fontsize=9)
ax.set_ylabel("Count", color='#aaa', fontsize=9)
style_ax(ax, f"Global Weight Distribution (n={len(all_w):,})")
stats_text = f"mean={np.mean(all_w):.6f}\nstd={np.std(all_w):.6f}\nmin={np.min(all_w):.4f}\nmax={np.max(all_w):.4f}"
ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=7,
        color='#ccc', ha='right', va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.8))

# ── 3b. Per-layer std dev (top N) ─────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1:])
layer_stats = []
for k, v in weights.items():
    if v.numel() > 100:  # skip tiny tensors
        layer_stats.append((k, v.float().std().item(), v.float().mean().item(), v.numel()))
layer_stats.sort(key=lambda x: -x[1])  # sort by std desc
top_n = min(40, len(layer_stats))
names = [s[0].replace("model.", "")[:50] for s in layer_stats[:top_n]]
stds = [s[1] for s in layer_stats[:top_n]]
means = [s[2] for s in layer_stats[:top_n]]

colors_bar = []
for s in layer_stats[:top_n]:
    k = s[0].lower()
    if any(x in k for x in ["action", "flow", "expert"]):
        colors_bar.append('#ff6b6b')
    elif any(x in k for x in ["vision", "image", "visual", "dino", "siglip"]):
        colors_bar.append('#ffd93d')
    else:
        colors_bar.append('#00d2ff')

ax.barh(range(top_n), stds, color=colors_bar, alpha=0.85, height=0.7)
ax.set_yticks(range(top_n))
ax.set_yticklabels(names, fontsize=6, color='#ccc', fontfamily='monospace')
ax.invert_yaxis()
ax.set_xlabel("Std Dev", color='#aaa', fontsize=9)
style_ax(ax, f"Per-Layer Std Dev (top {top_n}, sorted)")
# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#ff6b6b', label='Action/Flow'),
    Patch(facecolor='#ffd93d', label='Vision'),
    Patch(facecolor='#00d2ff', label='Language/Other'),
]
ax.legend(handles=legend_elements, fontsize=7, loc='lower right',
          facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')

# ── 3c. Parameter distribution pie ────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
categories = {}
for k, v in weights.items():
    kl = k.lower()
    if any(x in kl for x in ["action", "flow", "expert", "decoder"]):
        cat = "Action/Flow"
    elif any(x in kl for x in ["vision", "image", "visual", "dino", "siglip", "clip"]):
        cat = "Vision"
    elif any(x in kl for x in ["embed"]):
        cat = "Embedding"
    else:
        cat = "Language/Other"
    categories[cat] = categories.get(cat, 0) + v.numel()

pie_colors = {'Action/Flow': '#ff6b6b', 'Vision': '#ffd93d', 'Language/Other': '#00d2ff', 'Embedding': '#a78bfa'}
cols = [pie_colors.get(c, '#888') for c in categories.keys()]
wedges, texts, autotexts = ax.pie(
    categories.values(), labels=categories.keys(), colors=cols,
    autopct=lambda p: f'{p:.1f}%\n({int(p*sum(categories.values())/100):,})',
    textprops={'color': 'white', 'fontsize': 8},
    pctdistance=0.75
)
for at in autotexts:
    at.set_fontsize(6)
style_ax(ax, "Parameter Distribution by Module")

# ── 3d. Weight magnitude heatmaps for interesting layers ──────────────────────
def find_2d_weight(keys, prefer_keywords=None):
    """Find a 2D weight tensor from the given keys."""
    candidates = []
    for k in keys:
        v = weights[k]
        if v.dim() == 2 and v.shape[0] >= 16 and v.shape[1] >= 16:
            score = 0
            if prefer_keywords:
                for pw in prefer_keywords:
                    if pw in k.lower():
                        score += 1
            candidates.append((k, score, v.numel()))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (-x[1], -x[2]))
    k = candidates[0][0]
    return k, weights[k]

# Heatmap 1: largest action layer
ak, av = find_2d_weight(action_keys, ["linear", "weight", "proj"])
if ak is None:
    ak, av = find_2d_weight(list(weights.keys()), ["action", "flow"])

ax = fig.add_subplot(gs[1, 1])
if av is not None:
    data = av.float().numpy()
    h, w = data.shape
    show = data[:min(h, 128), :min(w, 128)]
    im = ax.imshow(show, cmap='RdBu_r', aspect='auto',
                   vmin=-np.percentile(np.abs(show), 98),
                   vmax=np.percentile(np.abs(show), 98))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    style_ax(ax, f"Action Weight Heatmap\n{ak[:60]}\n({h}×{w})")
else:
    ax.text(0.5, 0.5, "No 2D action weight found", transform=ax.transAxes,
            ha='center', va='center', color='#888')
    style_ax(ax, "Action Weight Heatmap")

# Heatmap 2: attention/language layer
lk, lv = find_2d_weight(language_keys + other_keys, ["attn", "q_proj", "k_proj", "self_attn"])
if lk is None:
    lk, lv = find_2d_weight(language_keys + other_keys, ["weight"])

ax = fig.add_subplot(gs[1, 2])
if lv is not None:
    data = lv.float().numpy()
    h, w = data.shape
    show = data[:min(h, 128), :min(w, 128)]
    im = ax.imshow(show, cmap='inferno', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    style_ax(ax, f"Language/Attn Heatmap\n{lk[:60]}\n({h}×{w})")
else:
    ax.text(0.5, 0.5, "No 2D language weight found", transform=ax.transAxes,
            ha='center', va='center', color='#888')
    style_ax(ax, "Language/Attn Heatmap")

# ── 3e. SVD spectrum of action head ──────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
svd_targets = action_keys if action_keys else list(weights.keys())
sk, sv = find_2d_weight(svd_targets, ["linear", "proj", "weight"])
if sv is not None:
    U, S, V = torch.svd(sv.float()[:min(sv.shape[0], 512), :min(sv.shape[1], 512)])
    s_np = S.numpy()
    ax.semilogy(s_np, color='#00ff88', linewidth=2)
    ax.fill_between(range(len(s_np)), s_np, alpha=0.15, color='#00ff88')
    ax.set_xlabel("Singular value index", color='#aaa', fontsize=9)
    ax.set_ylabel("Value (log)", color='#aaa', fontsize=9)
    # Effective rank (how many SVs needed for 90% energy)
    energy = np.cumsum(s_np**2) / np.sum(s_np**2)
    rank90 = np.searchsorted(energy, 0.9) + 1
    ax.axvline(rank90, color='#ff6b6b', linestyle='--', alpha=0.7)
    ax.text(rank90 + 2, s_np[0]*0.5, f'90% energy\nrank={rank90}',
            color='#ff6b6b', fontsize=8)
    style_ax(ax, f"SVD Spectrum\n{sk[:55]}")
else:
    ax.text(0.5, 0.5, "No suitable matrix found", transform=ax.transAxes,
            ha='center', va='center', color='#888')
    style_ax(ax, "SVD Spectrum")

# ── 3f. Layer-wise parameter count ───────────────────────────────────────────
ax = fig.add_subplot(gs[2, 1:])
# Group by top-2 level prefix
groups = {}
for k, v in weights.items():
    parts = k.split(".")
    prefix = ".".join(parts[:3]) if len(parts) >= 3 else k
    groups[prefix] = groups.get(prefix, 0) + v.numel()
# Show top 30
sorted_groups = sorted(groups.items(), key=lambda x: -x[1])[:30]
gnames = [g[0][:45] for g in sorted_groups]
gcounts = [g[1] for g in sorted_groups]
ax.barh(range(len(gnames)), gcounts, color='#a78bfa', alpha=0.8, height=0.7)
ax.set_yticks(range(len(gnames)))
ax.set_yticklabels(gnames, fontsize=6, color='#ccc', fontfamily='monospace')
ax.invert_yaxis()
ax.set_xlabel("Parameter count", color='#aaa', fontsize=9)
style_ax(ax, "Parameter Count by Layer Group (top 30)")

# ── 3g. Weight norm per layer (ordered) ──────────────────────────────────────
ax = fig.add_subplot(gs[3, :])
norms = []
for k, v in weights.items():
    if v.numel() > 100:
        norms.append((k, v.float().norm().item(), v.numel()))
# Keep original order
norm_vals = [n[1] for n in norms]
norm_names = [n[0].replace("model.", "")[:40] for n in norms]
ax.plot(norm_vals, color=ACCENT, linewidth=0.8, alpha=0.9)
ax.fill_between(range(len(norm_vals)), norm_vals, alpha=0.1, color=ACCENT)
# Mark top 5 outliers
top5_idx = np.argsort(norm_vals)[-5:]
for i in top5_idx:
    ax.annotate(norm_names[i], (i, norm_vals[i]),
                fontsize=5, color='#ff6b6b', rotation=30,
                ha='left', va='bottom')
    ax.plot(i, norm_vals[i], 'o', color='#ff6b6b', markersize=4)
ax.set_xlabel("Layer index (in storage order)", color='#aaa', fontsize=9)
ax.set_ylabel("Frobenius Norm", color='#aaa', fontsize=9)
style_ax(ax, "Weight Frobenius Norm per Layer")

# ── 3h. Sparsity analysis ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[4, 0])
sparsity = []
for k, v in weights.items():
    if v.numel() > 100:
        near_zero = (v.float().abs() < 1e-6).sum().item() / v.numel()
        sparsity.append((k, near_zero))
sparsity.sort(key=lambda x: -x[1])
top_sparse = sparsity[:20]
ax.barh(range(len(top_sparse)),
        [s[1]*100 for s in top_sparse],
        color='#ffd93d', alpha=0.8, height=0.7)
ax.set_yticks(range(len(top_sparse)))
ax.set_yticklabels([s[0].replace("model.","")[:40] for s in top_sparse],
                   fontsize=6, color='#ccc', fontfamily='monospace')
ax.invert_yaxis()
ax.set_xlabel("Near-zero weights (%)", color='#aaa', fontsize=9)
style_ax(ax, "Sparsity (top 20 layers, |w|<1e-6)")

# ── 3i. Quantization-readiness: per-layer dynamic range ─────────────────────
ax = fig.add_subplot(gs[4, 1])
dyn_range = []
for k, v in weights.items():
    if v.numel() > 100:
        vf = v.float()
        r = vf.max().item() - vf.min().item()
        dyn_range.append((k, r))
dyn_range.sort(key=lambda x: -x[1])
top_dr = dyn_range[:20]
ax.barh(range(len(top_dr)),
        [d[1] for d in top_dr],
        color='#ff6b6b', alpha=0.8, height=0.7)
ax.set_yticks(range(len(top_dr)))
ax.set_yticklabels([d[0].replace("model.","")[:40] for d in top_dr],
                   fontsize=6, color='#ccc', fontfamily='monospace')
ax.invert_yaxis()
ax.set_xlabel("Dynamic Range (max - min)", color='#aaa', fontsize=9)
style_ax(ax, "Dynamic Range (top 20) → Quantization Difficulty")

# ── 3j. Histogram comparison: vision vs language vs action ──────────────────
ax = fig.add_subplot(gs[4, 2])
for keys, label, color in [
    (vision_keys, "Vision", '#ffd93d'),
    (language_keys, "Language", '#00d2ff'),
    (action_keys, "Action", '#ff6b6b'),
]:
    if keys:
        vals = torch.cat([weights[k].float().flatten() for k in keys]).numpy()
        clip_v = np.percentile(np.abs(vals), 99)
        vals_c = vals[np.abs(vals) < clip_v]
        ax.hist(vals_c, bins=150, alpha=0.5, color=color, label=f"{label} ({len(keys)})",
                density=True, edgecolor='none')
if not (vision_keys or language_keys or action_keys):
    # Fallback: split all weights by index
    all_keys = list(weights.keys())
    third = len(all_keys) // 3
    for keys, label, color in [
        (all_keys[:third], "First 1/3", '#ffd93d'),
        (all_keys[third:2*third], "Middle 1/3", '#00d2ff'),
        (all_keys[2*third:], "Last 1/3", '#ff6b6b'),
    ]:
        vals = torch.cat([weights[k].float().flatten() for k in keys]).numpy()
        clip_v = np.percentile(np.abs(vals), 99)
        vals_c = vals[np.abs(vals) < clip_v]
        ax.hist(vals_c, bins=150, alpha=0.5, color=color, label=label,
                density=True, edgecolor='none')
ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')
ax.set_xlabel("Value", color='#aaa', fontsize=9)
style_ax(ax, "Weight Distribution by Module")

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = "smolvla_weights_visualization.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\n✅ Saved to {out_path}")
print("Open the image to explore your model's weight structure!")