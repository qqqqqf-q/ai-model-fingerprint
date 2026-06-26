#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据完整性检查
--------------
扫描 data/raw_*.jsonl, 对每个模型报告:
  - 总行数 / 有效(valid)数 / 错误数
  - 关键字段覆盖率: system_fingerprint / usage / response / returned_model
  - 文件大小
  - 损坏行数 (JSON 解析失败)
这样随时能确认 "没漏数据", 而不是凭感觉。

用法: python inspect.py
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def main():
    files = sorted(DATA_DIR.glob("raw_*.jsonl"))
    if not files:
        print("data/ 下还没有 raw_*.jsonl")
        return

    print(f"{'模型':<34}{'行':>6}{'有效':>6}{'错':>5}{'坏行':>5}"
          f"{'fp%':>6}{'usage%':>7}{'resp%':>6}{'retmod%':>8}{'大小':>9}")
    print("-" * 96)

    tot_lines = tot_valid = tot_bad = tot_size = 0
    for fp in files:
        name = fp.stem[len("raw_"):]
        lines = valid = bad = fp_ok = usage_ok = resp_ok = mod_ok = 0
        size = fp.stat().st_size
        with open(fp, encoding="utf-8") as f:
            for line in f:
                lines += 1
                try:
                    r = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                if r.get("valid"):
                    valid += 1
                if r.get("system_fingerprint"):
                    fp_ok += 1
                if r.get("usage"):
                    usage_ok += 1
                if r.get("response"):
                    resp_ok += 1
                if r.get("returned_model"):
                    mod_ok += 1
        tot_lines += lines; tot_valid += valid; tot_bad += bad; tot_size += size
        pct = lambda x: f"{100*x/max(1,lines):.0f}%"
        print(f"{name:<34}{lines:>6}{valid:>6}{lines-valid:>5}{bad:>5}"
              f"{pct(fp_ok):>6}{pct(usage_ok):>7}{pct(resp_ok):>6}{pct(mod_ok):>8}"
              f"{size/1024:>7.0f}K")

    print("-" * 96)
    print(f"合计: {len(files)} 个模型 | {tot_lines} 行 | 有效 {tot_valid} | "
          f"损坏行 {tot_bad} | {tot_size/1024/1024:.1f} MB")
    if tot_bad:
        print(f"⚠ 有 {tot_bad} 行损坏 (半行/被截断), 重跑会自动补采, 不影响有效样本计数")


if __name__ == "__main__":
    main()
