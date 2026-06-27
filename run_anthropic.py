#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anthropic Messages API 采集脚本 — 验证第三方渠道真假
--------------------------------------------------
对照基准: OpenRouter 官方渠道的同一模型指纹 (data/raw_anthropic_*.jsonl)。
若第三方渠道的指纹与官方基准高度一致 (cos→1), 则 likely 为真;
若分布明显偏移, 则 likely 掺假/换模型。

接口: POST {base_url}/v1/messages  (Anthropic 原生格式)
  headers: x-api-key, anthropic-version
  body: {model, max_tokens, temperature, messages}
  resp:  {content:[{text}], usage:{input_tokens,output_tokens}, model, stop_reason}

用法:
  python run_anthropic.py --base-url https://dasuapi.com \\
    --api-key sk-xxx --model claude-opus-4-6 --label dasuapi_claude-opus-4-6 \\
    --samples 300
"""
import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

PROMPT = "请从1到355之间随机选择一个数字，只输出这个数字，不要有任何其他内容。"
RANGE_MIN, RANGE_MAX = 1, 355


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def extract_number(text, lo, hi):
    if not text:
        return None
    m = re.search(r"\d+", text)
    if m:
        n = int(m.group())
        if lo <= n <= hi:
            return n
    return None


def _req(max_tokens):
    return {"prompt": PROMPT, "temperature": 1.0, "max_tokens": max_tokens,
            "range": [RANGE_MIN, RANGE_MAX], "api": "anthropic"}


async def fetch_one(session, base_url, api_key, model, prompt, max_tokens):
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {"Content-Type": "application/json",
               "x-api-key": api_key,
               "anthropic-version": "2023-06-01"}
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 1.0,
               "messages": [{"role": "user", "content": prompt}]}
    timeout = aiohttp.ClientTimeout(total=60)
    last_err = "exhausted"
    for attempt in range(4):
        try:
            async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                status = resp.status
                body = await resp.text()
                if status == 429 or status >= 500:
                    await asyncio.sleep(min(2 ** attempt, 16))
                    continue
                if status != 200:
                    return {"model": model, "requested_model": model, "returned_model": None,
                            "raw": None, "number": None, "valid": False, "error": f"http_{status}",
                            "http_status": status, "response": None, "response_body": body,
                            "usage": None, "system_fingerprint": None, "finish_reason": None,
                            "request": _req(max_tokens), "timestamp": time.time()}
                data = json.loads(body)
                content = ((data.get("content") or [{}])[0].get("text") or "").strip()
                num = extract_number(content, RANGE_MIN, RANGE_MAX)
                return {"model": model, "requested_model": model,
                        "returned_model": data.get("model"),
                        "raw": content, "number": num, "valid": num is not None,
                        "finish_reason": data.get("stop_reason"),
                        "usage": data.get("usage"), "system_fingerprint": None,
                        "request": _req(max_tokens), "response": data, "error": None,
                        "http_status": status, "timestamp": time.time()}
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            await asyncio.sleep(min(2 ** attempt, 16))
            last_err = repr(e)
        except Exception as e:
            last_err = repr(e)
            await asyncio.sleep(min(2 ** attempt, 16))
    return {"model": model, "requested_model": model, "returned_model": None,
            "raw": None, "number": None, "valid": False, "error": f"max_retries:{last_err}",
            "http_status": None, "response": None, "usage": None, "system_fingerprint": None,
            "finish_reason": None, "request": _req(max_tokens), "timestamp": time.time()}


def count_valid(fname):
    if not fname.exists():
        return 0
    n = 0
    with open(fname, encoding="utf-8") as f:
        for line in f:
            try:
                if json.loads(line).get("valid"):
                    n += 1
            except Exception:
                pass
    return n


async def collect(session, base_url, api_key, model, label, total, concurrency, max_tokens):
    fname = DATA_DIR / f"raw_{sanitize(label)}.jsonl"
    existing = count_valid(fname)
    if existing >= total:
        print(f"[{label}] ✓ 已有 {existing} 有效, 跳过")
        return
    print(f"[{label}] 已有 {existing}/{total}, 采集中 ...")
    sem = asyncio.Semaphore(concurrency)
    valid = existing
    in_flight = 0
    submitted = 0
    max_submit = total * 2 + concurrency * 4

    async def submit():
        nonlocal submitted
        submitted += 1
        async with sem:
            t0 = time.time()
            try:
                rec = await fetch_one(session, base_url, api_key, model, PROMPT, max_tokens)
            except Exception as e:
                rec = {"model": model, "requested_model": model, "raw": None, "number": None,
                       "valid": False, "error": f"submit_exc:{e!r}", "http_status": None,
                       "response": None, "usage": None, "system_fingerprint": None,
                       "finish_reason": None, "request": _req(max_tokens), "timestamp": time.time()}
            rec["latency"] = round(time.time() - t0, 3)
            return rec

    def launch():
        nonlocal in_flight
        in_flight += 1
        return asyncio.create_task(submit())

    f = open(fname, "a", encoding="utf-8")
    try:
        pending = {launch() for _ in range(min(concurrency, total - valid))}
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                in_flight -= 1
                try:
                    rec = d.result()
                except Exception as e:
                    rec = {"model": model, "valid": False, "error": f"task_exc:{e!r}",
                           "timestamp": time.time()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if rec.get("valid"):
                    valid += 1
                sys.stdout.write(f"\r[{label}] {valid}/{total} 有效 | 在飞 {in_flight} | 提交 {submitted}")
                sys.stdout.flush()
            if submitted < max_submit:
                slots = min(concurrency - in_flight, total - valid - in_flight)
                for _ in range(max(0, slots)):
                    pending.add(launch())
    finally:
        f.close()
    print(f"\n[{label}] ✓ 完成, 有效 {valid}/{total}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="数据文件名标签 (raw_{label}.jsonl)")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()
    print(f"== Anthropic Messages 采集 == {args.base_url} | model={args.model} | label={args.label}")
    print(f"prompt: {PROMPT}\n")
    connector = aiohttp.TCPConnector(limit=max(args.concurrency * 2, 32))
    async with aiohttp.ClientSession(connector=connector) as session:
        await collect(session, args.base_url, args.api_key, args.model, args.label,
                      args.samples, args.concurrency, args.max_tokens)
    print("\n完成. 运行 analyze.py 即可与 OpenRouter 基准对比指纹.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye")
