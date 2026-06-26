#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 模型指纹分析 + 可视化
----------------------
读取 data/raw_*.jsonl, 对每个模型:
  - 计算 1..RANGE 的频率分布指纹向量
  - 计算统计量 (mean/median/std/min/max/unique/mode)
  - 保留全部有效数字序列
两两模型间计算: 余弦相似度 / JS 散度 / overall = cosine*exp(-js)

输出:
  data/summary.json         全量汇总 (含指纹/统计量/相似度)
  figures/dist_overlay.*    分布叠加图
  figures/dist_grid.*       每模型分布子图
  figures/stats.*           统计量对比
  figures/similarity_*.png  相似度热力图 (cosine / JS)
  figures/boxplot.*         分布箱线图
  figures/topk.*            高频数字对比
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

RANGE_MIN, RANGE_MAX = 1, 355
K_BINS = 36  # 分布叠加图分箱数


def extract_number_v2(text, lo: int, hi: int):
    """改进的提取: 用 raw 重新解析, 恢复 minimax 等'重复输出数字'的样本。
    '142142' -> 142 (前半==后半且在范围内); 正常 '287' -> 287 不受影响。"""
    if not text:
        return None
    m = re.search(r"\d+", text)
    if not m:
        return None
    s = m.group()
    n = int(s)
    if lo <= n <= hi:
        return n
    # 重复模式: "142142" -> "142"
    if len(s) % 2 == 0:
        half = s[:len(s) // 2]
        if half * 2 == s:
            n2 = int(half)
            if lo <= n2 <= hi:
                return n2
    return None


def load_model(fname: Path):
    numbers, latencies, usages, raws, errors = [], [], [], [], 0
    if not fname.exists():
        return None
    with open(fname, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            # 用 raw 重新解析, 可恢复 minimax 重复数字; 正常模型结果不变
            num = extract_number_v2(r.get("raw"), RANGE_MIN, RANGE_MAX)
            if num is not None:
                numbers.append(num)
                if r.get("latency"):
                    latencies.append(r["latency"])
                if r.get("usage"):
                    usages.append(r["usage"])
                raws.append(r.get("raw"))
            elif r.get("error"):
                errors += 1
    if not numbers:
        return None
    return {"numbers": numbers, "latencies": latencies, "usages": usages,
            "raws": raws, "errors": errors}


def distribution(numbers):
    counts = np.zeros(RANGE_MAX, dtype=float)
    for n in numbers:
        if RANGE_MIN <= n <= RANGE_MAX:
            counts[n - 1] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def stats(numbers):
    arr = np.array(numbers, dtype=float)
    vals, cnts = np.unique(arr, return_counts=True)
    mode = int(vals[np.argmax(cnts)])
    return {
        "n": len(numbers),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "unique": int(len(vals)),
        "mode": mode,
        "mode_freq": int(cnts.max()),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
    }


def similarity(d1, d2):
    a = np.asarray(d1, dtype=float)
    b = np.asarray(d2, dtype=float)
    dot = float(np.dot(a, b))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    cosine = dot / (na * nb) if na > 0 and nb > 0 else 0.0
    eps = 1e-12
    p = a + eps
    q = b + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    js = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
    return {"cosine": cosine, "js": js, "overall": cosine * math.exp(-js)}


def main():
    files = sorted(DATA_DIR.glob("raw_*.jsonl"))
    if not files:
        sys.exit("✗ data/ 下没有 raw_*.jsonl, 先跑 run.py")

    models = {}  # name -> data
    for fp in files:
        name = fp.stem[len("raw_"):]
        m = load_model(fp)
        if m:
            models[name] = m

    if not models:
        sys.exit("✗ 没有读到任何有效样本")

    names = list(models.keys())
    print(f"读取 {len(names)} 个模型: {names}")

    # ---- 指纹 & 统计 ----
    summary = {"range": [RANGE_MIN, RANGE_MAX], "models": {}}
    dists = {}
    for name, m in models.items():
        d = distribution(m["numbers"])
        dists[name] = d
        s = stats(m["numbers"])
        usage_total = sum((u.get("total_tokens") or 0) for u in m["usages"])
        summary["models"][name] = {
            "stats": s,
            "errors": m["errors"],
            "total_tokens": usage_total,
            "mean_latency": float(np.mean(m["latencies"])) if m["latencies"] else None,
            "distribution": d.tolist(),
            "numbers": m["numbers"],  # 全量保留
        }
        print(f"  {name}: n={s['n']} mean={s['mean']:.1f} std={s['std']:.1f} "
              f"unique={s['unique']} mode={s['mode']}")

    # ---- 两两相似度 ----
    cos_m = np.zeros((len(names), len(names)))
    js_m = np.zeros_like(cos_m)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            sim = similarity(dists[a], dists[b])
            cos_m[i, j] = sim["cosine"]
            js_m[i, j] = sim["js"]
    summary["similarity"] = {
        "cosine": cos_m.tolist(), "js": js_m.tolist(), "labels": names}

    with open(DATA_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 汇总写入 {DATA_DIR/'summary.json'}")

    # ============ 画图 ============
    cmap = plt.get_cmap("tab20")

    # 1. 分布叠加 (分箱)
    fig, ax = plt.subplots(figsize=(14, 6))
    edges = np.linspace(RANGE_MIN, RANGE_MAX + 1, K_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for i, name in enumerate(names):
        hist, _ = np.histogram(models[name]["numbers"], bins=edges)
        hist = hist / hist.sum()
        ax.plot(centers, hist, drawstyle="steps-mid", label=name, color=cmap(i % 20))
    ax.set_xlabel("数字 (分箱)")
    ax.set_ylabel("频率")
    ax.set_title("各模型随机数字分布叠加 (temperature=1.0)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIG_DIR / "dist_overlay.png", dpi=140); plt.close(fig)

    # 2. 每模型分布子图
    ncol = min(3, len(names))
    nrow = math.ceil(len(names) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow), squeeze=False)
    for idx, name in enumerate(names):
        r, c = divmod(idx, ncol)
        ax = axes[r][c]
        ax.bar(centers, np.histogram(models[name]["numbers"], bins=edges)[0], width=centers[1]-centers[0], color=cmap(idx % 20))
        s = summary["models"][name]["stats"]
        ax.set_title(f"{name}\nmean={s['mean']:.1f} std={s['std']:.1f}", fontsize=9)
        ax.set_xlim(RANGE_MIN, RANGE_MAX)
    for idx in range(len(names), nrow * ncol):
        axes[divmod(idx, ncol)].axis("off")
    fig.tight_layout(); fig.savefig(FIG_DIR / "dist_grid.png", dpi=140); plt.close(fig)

    # 3. 统计量柱状图
    metric_names = ["mean", "median", "std", "unique"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for k, met in enumerate(metric_names):
        ax = axes[k // 2][k % 2]
        vals = [summary["models"][n]["stats"][met] for n in names]
        ax.bar(range(len(names)), vals, color=[cmap(i % 20) for i in range(len(names))])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(met)
    fig.tight_layout(); fig.savefig(FIG_DIR / "stats.png", dpi=140); plt.close(fig)

    # 4. 相似度热力图
    for mat, title, fname, cmapname in [
        (cos_m, "余弦相似度 (越高越像)", "similarity_cosine.png", "viridis"),
        (js_m, "JS 散度 (越低越像)", "similarity_js.png", "magma"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mat, cmap=cmapname, aspect="equal")
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7, color="white")
        ax.set_title(title)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=140); plt.close(fig)

    # 5. 箱线图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot([models[n]["numbers"] for n in names], tick_labels=names, showmeans=True)
    ax.set_ylabel("数字")
    ax.set_title("各模型分布箱线图")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "boxplot.png", dpi=140); plt.close(fig)

    # 6. Top-K 高频数字
    K = 15
    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.8 / len(names)
    all_nums = sorted({n for name in names for n in models[name]["numbers"]})
    # 取全局最高频 K 个数字
    global_cnt = {}
    for name in names:
        for n in models[name]["numbers"]:
            global_cnt[n] = global_cnt.get(n, 0) + 1
    topk = sorted(global_cnt, key=lambda x: -global_cnt[x])[:K]
    x = np.arange(len(topk))
    for i, name in enumerate(names):
        cnt = {n: 0 for n in topk}
        for n in models[name]["numbers"]:
            if n in cnt:
                cnt[n] += 1
        freq = [cnt[n] / max(1, summary["models"][name]["stats"]["n"]) for n in topk]
        ax.bar(x + i * width, freq, width, label=name, color=cmap(i % 20))
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels([str(n) for n in topk])
    ax.set_xlabel("高频数字 (全局 Top-15)")
    ax.set_ylabel("该模型内频率")
    ax.set_title("各模型高频数字对比")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIG_DIR / "topk.png", dpi=140); plt.close(fig)

    print(f"✓ 图表写入 {FIG_DIR}/  (dist_overlay, dist_grid, stats, similarity_*, boxplot, topk)")


if __name__ == "__main__":
    main()
