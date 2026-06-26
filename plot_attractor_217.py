#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""217 吸引子群: opus-4.6/4.7 (锁定) vs grok/gpt-5.x (分散) 分布对比。"""
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
group = [("anthropic_claude-opus-4.7", "opus-4.7", "#d62728"),
         ("anthropic_claude-opus-4.6", "opus-4.6", "#9467bd"),
         ("x-ai_grok-build-0.1", "grok-build-0.1", "#2ca02c"),
         ("openai_gpt-5.4", "gpt-5.4", "#1f77b4"),
         ("openai_gpt-5.5", "gpt-5.5", "#ff7f0e")]

agg = Counter()
for full, _, _ in group:
    for n in s["models"][full]["numbers"]:
        agg[n] += 1
top = sorted([n for n, _ in agg.most_common(15)])

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(top))
w = 0.8 / len(group)
for i, (full, name, c) in enumerate(group):
    nums = s["models"][full]["numbers"]
    freq = [nums.count(n) / len(nums) for n in top]
    f217 = nums.count(217) / len(nums)
    ax.bar(x + i * w - 0.4 + w / 2, freq, w,
           label=f"{name}  (217={f217:.0%}, unique={s['models'][full]['stats']['unique']})",
           color=c, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in top], fontsize=10)
ax.set_xlabel("数字 (群内合计 top-15)"); ax.set_ylabel("频率")
ax.set_title("217 吸引子群: opus-4.6/4.7 锁定 217 (82~96%), grok/gpt-5.x 仅弱偏 217\n"
             "grok↔opus-4.6=0.857 高相似, 但本质是 'opus 锁 217 + grok 偏 217' 的峰对齐", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "figures/attractor_217.png", dpi=150)
print("✓ figures/attractor_217.png")
