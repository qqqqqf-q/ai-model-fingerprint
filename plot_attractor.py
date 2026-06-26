#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""247 吸引子群: 5 个共享 247 偏好的模型, 分布形状对比。"""
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
group = [("anthropic_claude-haiku-4.5", "haiku-4.5", "#1f77b4"),
         ("anthropic_claude-sonnet-4.6", "sonnet-4.6", "#2ca02c"),
         ("qwen_qwen3.7-max", "qwen3.7-max", "#d62728"),
         ("z-ai_glm-5.2", "glm-5.2", "#ff7f0e"),
         ("moonshotai_kimi-k2.7-code", "kimi-k2.7-code", "#9467bd")]

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
    ax.bar(x + i * w - 0.4 + w / 2, freq, w,
           label=f"{name}  (247={nums.count(247)/len(nums):.0%}, unique={s['models'][full]['stats']['unique']})",
           color=c, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in top], fontsize=10)
ax.set_xlabel("数字 (群内合计 top-15)"); ax.set_ylabel("频率")
ax.set_title("247 吸引子群: 5 个模型共享 247 峰值, 但分布形状各异\n247 是共同峰值; 第二/三高频数字不同 → 非整体分布继承", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "figures/attractor_247.png", dpi=150)
print("✓ figures/attractor_247.png")
