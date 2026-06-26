#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token / Cost 用量预估
---------------------
读取已采集 data/raw_*.jsonl 的真实 usage (prompt_tokens / completion_tokens / cost),
外推到 config 的全量目标 samples_per_model。
- 已采模型: 用自身实测均值外推 (准)
- 未采模型: 用所有已采模型均值外推 (粗估, 标 *, 实测后自动变准)

用法: python estimate.py [--samples N] [--models a b c]
"""
import argparse
import json
import re
import statistics
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def usage_of(rec):
    u = rec.get("usage") or {}
    p = u.get("prompt_tokens") or u.get("input_tokens") or 0
    c = u.get("completion_tokens") or u.get("output_tokens") or 0
    cost = u.get("cost") or 0
    return p, c, cost


def model_stats(name):
    fp = DATA_DIR / f"raw_{sanitize(name)}.jsonl"
    if not fp.exists():
        return None
    ps, cs, costs = [], [], []
    with open(fp, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("valid"):
                p, c, cost = usage_of(r)
                if p or c:
                    ps.append(p); cs.append(c); costs.append(cost)
    if not ps:
        return None
    return {"n": len(ps), "in_mean": statistics.mean(ps),
            "out_mean": statistics.mean(cs), "cost_mean": statistics.mean(costs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--samples", type=int)
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    target = args.samples or cfg["sampling"]["samples_per_model"]
    models = args.models or cfg["models"]

    all_in, all_out, all_cost = [], [], []
    rows = []
    for m in models:
        s = model_stats(m)
        if s:
            all_in.append(s["in_mean"]); all_out.append(s["out_mean"]); all_cost.append(s["cost_mean"])
            rows.append((m, s, "实测"))
        else:
            rows.append((m, None, "未采"))
    fb_in = statistics.mean(all_in) if all_in else 28.0
    fb_out = statistics.mean(all_out) if all_out else 1.5
    fb_cost = statistics.mean(all_cost) if all_cost else 0.0

    print(f"全量目标: 每模型 {target} 次 × {len(models)} 模型 = {len(models)*target} 请求\n")
    print(f"{'模型':<34}{'已采':>5}{'in/req':>9}{'out/req':>9}{'$/req':>10}   来源")
    print("-" * 86)
    tot_in = tot_out = tot_cost = 0.0
    for m, s, src in rows:
        if s:
            in_per, out_per, cost_per, n = s["in_mean"], s["out_mean"], s["cost_mean"], s["n"]
        else:
            in_per, out_per, cost_per, n = fb_in, fb_out, fb_cost, 0
            src = "未采*(全局均值)"
        tot_in += in_per * target
        tot_out += out_per * target
        tot_cost += cost_per * target
        print(f"{m:<34}{n:>5}{in_per:>9.1f}{out_per:>9.1f}{cost_per:>10.6f}   {src}")

    print("-" * 86)
    print(f"全量预估 (每模型 {target} 次):")
    print(f"  总 input   ≈ {tot_in:>12,.0f} token")
    print(f"  总 output  ≈ {tot_out:>12,.0f} token")
    print(f"  总 token   ≈ {tot_in+tot_out:>12,.0f}")
    print(f"  总 cost    ≈ ${tot_cost:.4f}")
    if all_cost:
        print(f"\n注: cost 仅对已采模型({len(all_cost)}个)准确; 未采模型用全局均值(*), "
              f"贵模型(opus/gpt-5.x)实际 cost 会更高, 全量跑完后此值自动变准。")
        # 保守上界: 按 5x 均值给个量级
        print(f"保守上界参考(×5): ≈ ${tot_cost*5:.2f}")


if __name__ == "__main__":
    main()
