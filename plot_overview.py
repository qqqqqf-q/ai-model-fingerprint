#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""综合分析图: 一张图讲完 17 个模型的指纹发现。"""
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
models = s["models"]
labels = s["similarity"]["labels"]
cos = np.array(s["similarity"]["cosine"])


def nums(m):
    return np.array(models[m]["numbers"])


fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# A: claude opus 4.6/4.7/4.8 分布对比
ax = axes[0, 0]
edges = np.linspace(1, 356, 36)
cent = 0.5 * (edges[:-1] + edges[1:])
for m, c in [("anthropic_claude-opus-4.6", "#2ca02c"),
             ("anthropic_claude-opus-4.7", "#1f77b4"),
             ("anthropic_claude-opus-4.8", "#d62728")]:
    h, _ = np.histogram(nums(m), bins=edges)
    st = models[m]["stats"]
    ax.plot(cent, h / h.sum(), drawstyle="steps-mid", color=c, lw=2,
            label=f"{m.split('_')[-1]}  (unique={st['unique']}, std={st['std']:.1f}, mean={st['mean']:.0f})")
ax.set_title("① Claude Opus 4.6/4.7/4.8 随机数分布\n4.6≈4.7(同指纹); 4.8 退化为准固定输出(unique=2)", fontsize=11)
ax.set_xlabel("数字"); ax.set_ylabel("频率"); ax.legend(fontsize=9); ax.set_xlim(1, 355)

# B: 相似度热力图
ax = axes[0, 1]
short = [l.split("_", 1)[-1] for l in labels]
im = ax.imshow(cos, cmap="viridis", aspect="equal", vmin=0, vmax=1)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(short, rotation=75, fontsize=7)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(short, fontsize=7)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                color="white" if cos[i, j] < 0.7 else "black")
ax.set_title("② 两两余弦相似度 (越亮=指纹越像)\n对角线=1; 块状结构揭示同族指纹", fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# C: 同栈对照 箱线图
ax = axes[1, 0]
pairs = [("moonshotai_kimi-k2.6", "kimi-k2.6"),
         ("moonshotai_kimi-k2.7-code", "kimi-k2.7-code"),
         ("deepseek_deepseek-v3.2", "ds-v3.2"),
         ("deepseek_deepseek-v4-flash", "ds-v4-flash"),
         ("deepseek_deepseek-v4-pro", "ds-v4-pro")]
data = [nums(full) for full, _ in pairs]
ticks = [s_ for _, s_ in pairs]
bp = ax.boxplot(data, vert=True, patch_artist=True, showmeans=True)
for patch, c in zip(bp["boxes"], ["#9467bd", "#8c564b", "#1f77b4", "#ff7f0e", "#2ca02c"]):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.set_xticklabels(ticks, fontsize=9)
ax.set_ylabel("数字")
ax.set_title("③ 同栈对照\nkimi k2.6→k2.7-code(同base后训练) / deepseek v3.2/v4-flash/v4-pro(同栈不同语料)", fontsize=11)

# D: mean & unique
ax = axes[1, 1]
order = sorted(models.keys(), key=lambda m: models[m]["stats"]["mean"])
names = [m.split("_", 1)[-1] for m in order]
means = [models[m]["stats"]["mean"] for m in order]
uniques = [models[m]["stats"]["unique"] for m in order]
x = np.arange(len(order))
ax.bar(x - 0.2, means, 0.4, label="均值 mean", color="#4c72b0")
ax2 = ax.twinx()
ax2.bar(x + 0.2, uniques, 0.4, label="多样性 unique", color="#dd8452")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=75, fontsize=7)
ax.set_ylabel("均值", color="#4c72b0"); ax2.set_ylabel("不同数字数 unique", color="#dd8452")
ax.set_title("④ 各模型均值 & 输出多样性\nunique 低=分布集中=确定性高 (opus-4.8=2 极端)", fontsize=11)
ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)

fig.suptitle("17 个 AI 模型随机数指纹分析  (OpenRouter, temperature=1.0, 每模型 ~300 样本)",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(ROOT / "figures/overview.png", dpi=150)
print("✓ figures/overview.png")
