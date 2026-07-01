#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dasuapi 专属 overview: 4 子图 (overview 风格参数), 主角 dasuapi 掺假验证。"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
s = json.load(open(ROOT / "data/summary.json"))
models = s["models"]
labels = s["similarity"]["labels"]
cos = np.array(s["similarity"]["cosine"])
DASU = "dasuapi_claude-opus-4-6"
OFFI = "anthropic_claude-opus-4.6"
GEM1 = "google_gemini-3.5-flash"
GEM2 = "google_gemini-3-flash-preview"


def nums(m):
    return np.array(models[m]["numbers"])


fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# ① dasuapi vs 官方 opus-4.6 分布
ax = axes[0, 0]
edges = np.linspace(1, 356, 36)
cent = 0.5 * (edges[:-1] + edges[1:])
for m, c, lab in [(DASU, "#d62728", "dasuapi 'claude-opus-4-6' (假)"),
                  (OFFI, "#1f77b4", "官方 claude-opus-4.6 (真)")]:
    h, _ = np.histogram(nums(m), bins=edges)
    st = models[m]["stats"]
    ax.plot(cent, h / h.sum(), drawstyle="steps-mid", color=c, lw=2.5,
            label=f"{lab}  mode={st['mode']}({st['distribution'] if False else ''})")
ax.set_title("① dasuapi vs 官方 opus-4.6 分布\n真品锁 217(82%), dasuapi 锁 187(51%) — 完全不同数字", fontsize=12)
ax.set_xlabel("数字"); ax.set_ylabel("频率"); ax.legend(fontsize=10); ax.set_xlim(1, 355)

# ② dasuapi 与所有模型相似度 (横向柱, overview 字号)
ax = axes[0, 1]
i = labels.index(DASU)
pairs = sorted([(cos[i][k], labels[k]) for k in range(len(labels)) if k != i], reverse=True)
names = [p[1].split("_", 1)[1] for p in pairs]
vals = [p[0] for p in pairs]


def color(n):
    if n == "claude-opus-4.6":
        return "#d62728"
    if "gemini" in n:
        return "#2ca02c"
    return "#999999"


bars = ax.barh(range(len(names)), vals, color=[color(n) for n in names])
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("余弦相似度")
ax.set_title("② dasuapi 与各模型余弦相似度\n红=官方opus-4.6(应最高, 实际0.21)  绿=gemini系(反而0.70/0.63)", fontsize=12)
ax.axvline(0.95, color="red", ls="--", lw=1.5, alpha=0.6, label="真品阈值~0.95")
for idx, n in enumerate(names):
    if n == "claude-opus-4.6" or "gemini" in n:
        ax.text(vals[idx] + 0.015, idx, f"{vals[idx]:.3f}", va="center", fontsize=10, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)

# ③ 四模型箱线图
ax = axes[1, 0]
box_data = [nums(DASU), nums(OFFI), nums(GEM1), nums(GEM2)]
box_names = ["dasuapi\n(假)", "官方opus-4.6\n(真)", "gemini-3.5-flash", "gemini-3-flash-preview"]
bp = ax.boxplot(box_data, tick_labels=box_names, patch_artist=True, showmeans=True)
for patch, c in zip(bp["boxes"], ["#d62728", "#1f77b4", "#2ca02c", "#2ca02c"]):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.set_ylabel("数字")
ax.set_title("③ 分布箱线图\ndasuapi 形状接近 gemini 系, 远离官方 opus-4.6", fontsize=12)
plt.setp(ax.get_xticklabels(), fontsize=10)

# ④ dasuapi vs opus-4.6 高频数字对比
ax = axes[1, 1]
agg = Counter()
for m in [DASU, OFFI]:
    for n in nums(m):
        agg[n] += 1
top = sorted([n for n, _ in agg.most_common(12)])
x = np.arange(len(top)); w = 0.4
for k, (m, c, lab) in enumerate([(DASU, "#d62728", "dasuapi(假)"), (OFFI, "#1f77b4", "官方opus-4.6(真)")]):
    ns = nums(m)
    freq = [int(np.sum(ns == n)) / len(ns) for n in top]
    ax.bar(x + (k - 0.5) * w, freq, w, label=lab, color=c, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([str(n) for n in top], fontsize=11)
ax.set_xlabel("数字 (两者合计 top-12)"); ax.set_ylabel("频率")
ax.set_title("④ 高频数字对比\ndasuapi 峰在 187/247; 官方 opus-4.6 峰在 217 — 几乎不重叠", fontsize=12)
ax.legend(fontsize=10)

fig.suptitle("dasuapi 'claude-opus-4-6' 掺假验证  (vs OpenRouter 官方基准, 300 样本/模型)",
             fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(ROOT / "figures/dasuapi_overview.png", dpi=150)
print("✓ figures/dasuapi_overview.png")
