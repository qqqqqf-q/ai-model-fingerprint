#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单独的相似度方阵热力图 (对角线=1), 含 dasuapi 作为第 23 个模型。大尺寸, 数字对齐。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
s = json.load(open(ROOT / "data/summary.json"))
labels = s["similarity"]["labels"]
cos = np.array(s["similarity"]["cosine"])
def short_name(l):
    if l.startswith("dasuapi_"):
        return "【dasuapi】" + l[len("dasuapi_"):]
    return l.split("_", 1)[-1]


short = [short_name(l) for l in labels]

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cos, cmap="viridis", aspect="equal", vmin=0, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(short, rotation=75, fontsize=9)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(short, fontsize=9)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if cos[i, j] < 0.7 else "black")
ax.set_title(f"{len(labels)} 个模型两两余弦相似度 (对角线=1, 越亮越像)", fontsize=14)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(ROOT / "figures/similarity_cosine.png", dpi=150)
print("✓ figures/similarity_cosine.png")
