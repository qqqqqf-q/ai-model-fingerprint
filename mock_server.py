#!/usr/bin/env python3
"""本地 mock 服务器: 模拟 OpenRouter 的 /chat/completions, 每个请求 sleep 0.3s"""
import asyncio, random, time
from aiohttp import web

async def chat(request):
    await asyncio.sleep(0.3)
    n = random.randint(1, 355)
    return web.json_response({
        "id": "mock-" + str(time.time()),
        "model": "mock/test",
        "choices": [{"message": {"content": str(n)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 1, "total_tokens": 51, "cost": 0},
        "system_fingerprint": "fp_mock_123",
    })

async def main():
    app = web.Application()
    app.router.add_post("/chat/completions", chat)
    runner = web.AppRunner(app); await runner.setup()
    site = web.TCPSite(runner, "localhost", 8765); await site.start()
    print("mock server on http://localhost:8765", flush=True)
    await asyncio.Event().wait()

asyncio.run(main())
