#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 模型指纹采集脚本
------------------
原理: 大模型并非真随机数生成器。固定 prompt "请从1到355之间随机选一个数字",
      在 temperature=1.0 下大量采样, 不同模型会形成统计上可区分的分布指纹。

本脚本:
  - 对 config.yaml 中每个模型并发采样 N 个有效样本
  - 把每次请求的原始返回 / 提取数字 / 延迟 / token 用量 全量落盘 (JSONL)
  - 支持断点续采 (中断后再跑会跳过已采集的有效样本)
  - 429 / 网络错误指数退避重试

用法:
  python run.py                         # 用 config.yaml
  python run.py --samples 500           # 覆盖采样数
  python run.py --models "a/b" "c/d"    # 覆盖模型列表
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import aiohttp
import yaml

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    api = cfg["api"]
    if not api.get("api_key"):
        api["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
    if not api["api_key"]:
        sys.exit("✗ 未找到 api_key: 请在 config.yaml 填写或设置 OPENROUTER_API_KEY")
    if not cfg.get("models"):
        sys.exit("✗ config.yaml 中 models 列表为空")
    return cfg


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def make_prompt(lo: int, hi: int) -> str:
    return f"请从{lo}到{hi}之间随机选择一个数字，只输出这个数字，不要有任何其他内容。"


def extract_number(text: str, lo: int, hi: int):
    if not text:
        return None
    m = re.search(r"\d+", text)
    if m:
        n = int(m.group())
        if lo <= n <= hi:
            return n
    return None


def _request_meta(prompt: str, samp_cfg: dict, model: str) -> dict:
    """记录请求参数, 保证可复现"""
    ov = samp_cfg.get("model_overrides", {}).get(model, {})
    return {
        "prompt": prompt,
        "temperature": 1.0,
        "max_tokens": ov.get("max_tokens", samp_cfg.get("max_tokens", 32)),
        "range": [samp_cfg["range_min"], samp_cfg["range_max"]],
        "reasoning_disabled": ov.get("reasoning_disabled", True),
    }


async def fetch_one(session, api_cfg, samp_cfg, model: str, prompt: str):
    """发一次请求, 返回完整记录 dict (可能 valid=False)"""
    url = api_cfg["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_cfg['api_key']}"}
    if api_cfg.get("http_referer"):
        headers["HTTP-Referer"] = api_cfg["http_referer"]
    if api_cfg.get("x_title"):
        headers["X-Title"] = api_cfg["x_title"]

    ov = samp_cfg.get("model_overrides", {}).get(model, {})
    max_tokens = ov.get("max_tokens", samp_cfg.get("max_tokens", 32))
    reasoning_disabled = ov.get("reasoning_disabled", True)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "max_tokens": max_tokens,
    }
    if reasoning_disabled:
        payload["reasoning"] = {"enabled": False}  # 禁用 thinking/推理, 避免 content 被截断空
    timeout = aiohttp.ClientTimeout(total=samp_cfg.get("request_timeout", 60))
    max_retries = samp_cfg.get("max_retries", 4)
    last_err = "exhausted"

    for attempt in range(max_retries):
        try:
            async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                status = resp.status
                body = await resp.text()
                if status == 429 or status >= 500:
                    # 退避后重试
                    await asyncio.sleep(min(2 ** attempt, 16))
                    continue
                if status != 200:
                    return {
                        "model": model, "requested_model": model, "returned_model": None,
                        "raw": None, "number": None, "valid": False,
                        "error": f"http_{status}", "http_status": status,
                        "response": None, "response_body": body,
                        "usage": None, "system_fingerprint": None, "finish_reason": None,
                        "request": _request_meta(prompt, samp_cfg, model),
                        "timestamp": time.time(),
                    }
                data = json.loads(body)
                choice = (data.get("choices") or [{}])[0]
                content = (choice.get("message") or {}).get("content", "")
                content = (content or "").strip()
                num = extract_number(content, samp_cfg["range_min"], samp_cfg["range_max"])
                # 关键: 存完整 response body, 绝不漏字段 (system_fingerprint/returned_model/cost/...)
                return {
                    "model": model,
                    "requested_model": model,
                    "returned_model": data.get("model"),
                    "id": data.get("id"),
                    "system_fingerprint": data.get("system_fingerprint"),
                    "raw": content,
                    "number": num,
                    "valid": num is not None,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": data.get("usage"),
                    "request": _request_meta(prompt, samp_cfg, model),
                    "response": data,            # 完整原始 response body
                    "error": None,
                    "http_status": status,
                    "timestamp": time.time(),
                }
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            await asyncio.sleep(min(2 ** attempt, 16))
            last_err = repr(e)
        except Exception as e:
            last_err = repr(e)
            await asyncio.sleep(min(2 ** attempt, 16))
    return {
        "model": model, "requested_model": model, "returned_model": None,
        "raw": None, "number": None, "valid": False,
        "error": f"max_retries:{last_err}", "http_status": None,
        "response": None, "response_body": None,
        "usage": None, "system_fingerprint": None, "finish_reason": None,
        "request": _request_meta(prompt, samp_cfg, model),
        "timestamp": time.time(),
    }


def count_valid(fname: Path) -> int:
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


async def collect_model(session, api_cfg, samp_cfg, model: str, total: int, prompt: str):
    fname = DATA_DIR / f"raw_{sanitize(model)}.jsonl"
    existing = count_valid(fname)
    needed = total - existing
    if needed <= 0:
        print(f"[{model}] ✓ 已有 {existing} 有效样本, 跳过")
        return existing

    print(f"[{model}] 已有 {existing}/{total}, 采集中 ...")
    sem = asyncio.Semaphore(samp_cfg["concurrency"])
    valid = existing
    in_flight = 0
    submitted = 0
    max_submit = total * 2 + samp_cfg["concurrency"] * 4  # 失败兜底上限

    async def submit():
        nonlocal submitted
        submitted += 1
        async with sem:
            t0 = time.time()
            try:
                rec = await fetch_one(session, api_cfg, samp_cfg, model, prompt)
            except Exception as e:
                # 任何意外都不丢: 记录一条 error 行, 不让单次异常炸掉整个采集
                rec = {
                    "model": model, "requested_model": model, "returned_model": None,
                    "raw": None, "number": None, "valid": False,
                    "error": f"submit_exc:{e!r}", "http_status": None,
                    "response": None, "usage": None,
                    "system_fingerprint": None, "finish_reason": None,
                    "request": _request_meta(prompt, samp_cfg, model),
                    "timestamp": time.time(),
                }
            rec["latency"] = round(time.time() - t0, 3)
            return rec

    def launch():
        nonlocal in_flight
        in_flight += 1
        return asyncio.create_task(submit())

    f = open(fname, "a", encoding="utf-8")
    try:
        pending = {launch() for _ in range(min(samp_cfg["concurrency"], total - valid))}
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                in_flight -= 1
                try:
                    rec = d.result()
                except Exception as e:
                    rec = {"model": model, "requested_model": model, "valid": False,
                           "error": f"task_exc:{e!r}", "http_status": None,
                           "response": None, "usage": None, "raw": None, "number": None,
                           "timestamp": time.time()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if samp_cfg.get("fsync"):
                    os.fsync(f.fileno())
                if rec.get("valid"):
                    valid += 1
                sys.stdout.write(f"\r[{model}] {valid}/{total} 有效 | 在飞 {in_flight} | 提交 {submitted}")
                sys.stdout.flush()
            # 一批处理完统一补充: 维持并发, 但 (valid+在飞) 不超过 total, 杜绝超采
            if submitted < max_submit:
                slots = min(samp_cfg["concurrency"] - in_flight, total - valid - in_flight)
                for _ in range(max(0, slots)):
                    pending.add(launch())
    finally:
        f.close()
    print(f"\n[{model}] ✓ 完成, 有效 {valid}/{total}")
    return valid


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--samples", type=int, help="覆盖 samples_per_model")
    ap.add_argument("--models", nargs="+", help="覆盖模型列表")
    ap.add_argument("--concurrency", type=int, help="覆盖并发数 (默认 8)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    samp = cfg["sampling"]
    if args.samples:
        samp["samples_per_model"] = args.samples
    if args.concurrency:
        samp["concurrency"] = args.concurrency
    models = args.models or cfg["models"]
    prompt = make_prompt(samp["range_min"], samp["range_max"])

    print(f"== 指纹采集 == base_url={cfg['api']['base_url']}")
    print(f"模型 {len(models)} 个, 每个采样 {samp['samples_per_model']}, 并发 {samp['concurrency']}")
    print(f"prompt: {prompt}\n")

    connector = aiohttp.TCPConnector(limit=max(samp["concurrency"] * 2, 32))
    async with aiohttp.ClientSession(connector=connector) as session:
        for model in models:
            try:
                await collect_model(session, cfg["api"], samp, model,
                                    samp["samples_per_model"], prompt)
            except KeyboardInterrupt:
                print("\n中断, 已保存的数据保留, 可重跑续采")
                raise
            except Exception as e:
                print(f"\n[{model}] 采集异常: {e}")
    print("\n全部完成. 运行 python analyze.py 生成统计与图表.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye")
