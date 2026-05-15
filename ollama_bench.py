#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------------------
# ollama_bench : A quick hacky script, inspired by ollama_bench over at https://github.com/dkruyt/ollama_bench
# specifically created to test ollama_proxy, and workloads typical to the VibeBuddy64U Project
#
# https://github.com/brett-olsen/ollama-proxy
# Version 0.2
# Created by Brett Olsen - 2026
# ----------------------------------------------------------------------------------------------------------------------

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field

import httpx

OLLAMA_BASE = "http://127.0.0.1:11434"

USAGE_EXAMPLE = """\
┌─────────────────────────────────────────────────────────┐
│              ollama_bench.py — quick start              │
└─────────────────────────────────────────────────────────┘

Required flags
  --models      Model tag to benchmark
  --requests    Total number of requests to send
  --concurrency How many requests run in parallel
  --chat        Use /api/chat  (omit for /api/generate)
  --think       on | off  — enable / disable thinking mode
  --context     Context window size in tokens
  --system      System prompt (chat mode only)
  --prompt      User prompt
  --warmup      Throwaway requests before benchmarking (default: 0)

Ready-to-run example:

  python ollama_bench.py \\
    --models gemma4:26b \\
    --requests 1 \\
    --concurrency 1 \\
    --chat \\
    --think off \\
    --context 104448 \\
    --warmup 1 \\
    --system "You are a helpful AI assistant being accessed from a real Commodore 64 home computer through a network bridge. The user is reading your reply on a 40-column text screen with very limited memory." \\
    --prompt "Explain in great detail, 6502 Machine Language, with examples, also explore the limitations of machine language coding on the Commodore 64"
"""


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class Result:
    model: str
    ok: bool
    elapsed: float = 0.0
    tokens_prompt: int = 0
    tokens_eval: int = 0
    text: str = ""
    error: str = ""
    tps: float = field(init=False)

    def __post_init__(self):
        self.tps = self.tokens_eval / self.elapsed if self.elapsed > 0 else 0.0


# ── single request ────────────────────────────────────────────────────────────

async def run_request(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    system: str,
    chat: bool,
    think: bool,
) -> Result:
    start = time.perf_counter()
    text_chunks: list[str] = []
    tokens_prompt = tokens_eval = 0

    try:
        if chat:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model":      model,
                "messages":   messages,
                "stream":     True,
                "think":      bool(think),
                "keep_alive": -1,
            }
            endpoint = "/api/chat"
        else:
            payload = {
                "model":      model,
                "prompt":     prompt,
                "system":     system,
                "stream":     True,
                "think":      bool(think),
                "keep_alive": -1,
            }
            endpoint = "/api/generate"

        #print(f"\n  [debug] payload → {json.dumps(payload, indent=2)}\n", flush=True)

        async with client.stream("POST", endpoint, json=payload, timeout=600) as r:
            r.raise_for_status()
            async for raw in r.aiter_lines():
                if not raw.strip():
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if chat:
                    text_chunks.append(chunk.get("message", {}).get("content") or "")
                else:
                    text_chunks.append(chunk.get("response") or "")

                if chunk.get("done"):
                    tokens_prompt = chunk.get("prompt_eval_count", 0)
                    tokens_eval   = chunk.get("eval_count", 0)
                    break

    except Exception as exc:
        return Result(model=model, ok=False, elapsed=time.perf_counter() - start,
                      error=str(exc))

    return Result(
        model=model,
        ok=True,
        elapsed=time.perf_counter() - start,
        tokens_prompt=tokens_prompt,
        tokens_eval=tokens_eval,
        text="".join(text_chunks),
    )


# ── warmup ────────────────────────────────────────────────────────────────────

async def run_warmup(
    model: str,
    prompt: str,
    system: str,
    chat: bool,
    think: bool,
    count: int,
) -> None:
    if count <= 0:
        return
    print(f"\n  Warming up — {count} request(s), results discarded…", flush=True)
    async with httpx.AsyncClient(base_url=OLLAMA_BASE) as client:
        for i in range(count):
            print(f"  [warmup {i + 1}/{count}] sending…", flush=True)
            r = await run_request(client, model, prompt, system, chat, think)
            status = f"{r.elapsed:.1f}s" if r.ok else f"ERROR: {r.error}"
            print(f"  [warmup {i + 1}/{count}] done — {status}", flush=True)
    print("  Warmup complete, starting benchmark…\n", flush=True)


# ── concurrent batch ──────────────────────────────────────────────────────────

async def run_batch(
    model: str,
    prompt: str,
    system: str,
    chat: bool,
    think: bool,
    total: int,
    concurrency: int,
) -> list[Result]:
    sem = asyncio.Semaphore(concurrency)
    results: list[Result] = []

    async def bounded(idx: int):
        async with sem:
            print(f"  [{idx + 1}/{total}] sending…", flush=True)
            async with httpx.AsyncClient(base_url=OLLAMA_BASE) as client:
                r = await run_request(client, model, prompt, system, chat, think)
            status = f"{r.elapsed:.1f}s  {r.tps:.1f} t/s" if r.ok else f"ERROR: {r.error}"
            print(f"  [{idx + 1}/{total}] done — {status}", flush=True)
            results.append(r)

    await asyncio.gather(*[bounded(i) for i in range(total)])
    return results


# ── summary ───────────────────────────────────────────────────────────────────

def print_table(results: list[Result]) -> None:
    col = {"#": 4, "status": 7, "latency": 10, "tok_in": 8, "tok_out": 9, "t/s": 8}
    header = (
        f"  {'#':>{col['#']}}  "
        f"{'status':<{col['status']}}  "
        f"{'latency':>{col['latency']}}  "
        f"{'tok_in':>{col['tok_in']}}  "
        f"{'tok_out':>{col['tok_out']}}  "
        f"{'t/s':>{col['t/s']}}"
    )
    divider = "  " + "─" * (len(header) - 2)

    print("\n── Results ──────────────────────────────────────────")
    print(header)
    print(divider)

    for i, r in enumerate(results, 1):
        if r.ok:
            print(
                f"  {i:>{col['#']}}  "
                f"{'✓ ok':<{col['status']}}  "
                f"{r.elapsed:>{col['latency'] - 1}.2f}s  "
                f"{r.tokens_prompt:>{col['tok_in']}}  "
                f"{r.tokens_eval:>{col['tok_out']}}  "
                f"{r.tps:>{col['t/s'] - 4}.1f} t/s"
            )
        else:
            short_err = r.error[:30] + "…" if len(r.error) > 30 else r.error
            print(
                f"  {i:>{col['#']}}  "
                f"{'✗ err':<{col['status']}}  "
                f"  {short_err}"
            )

    print(divider)


def _trunc(text: str, width: int) -> str:
    return text if len(text) <= width else text[:width - 1] + "…"


def print_summary(
    model: str,
    results: list[Result],
    think: bool,
    context: int,
    mode: str,
    concurrency: int,
    warmup: int,
    system: str,
    prompt: str,
) -> None:
    good = [r for r in results if r.ok]
    bad  = [r for r in results if not r.ok]

    W = 55  # ruler width
    print("\n" + "─" * W)
    print(f"  Model       : {model}")
    print(f"  Mode        : {mode}")
    print(f"  Think       : {'on' if think else 'off'}")
    print(f"  Context     : {context:,} tokens")
    print(f"  Requests    : {len(results)}  ✓ {len(good)}  ✗ {len(bad)}  concurrency: {concurrency}")
    print(f"  Warmup      : {warmup} pass{'es' if warmup != 1 else ''}")
    print(f"  System      : {_trunc(system or '(none)', W - 14)}")
    print(f"  Prompt      : {_trunc(prompt, W - 14)}")

    if good:
        times      = [r.elapsed for r in good]
        speeds     = [r.tps for r in good]
        total_eval = sum(r.tokens_eval for r in good)

        print(f"  Latency     : avg {sum(times)/len(times):.2f}s  "
              f"min {min(times):.2f}s  max {max(times):.2f}s")
        print(f"  Throughput  : avg {sum(speeds)/len(speeds):.1f} t/s  "
              f"max {max(speeds):.1f} t/s")
        print(f"  Tokens out  : {total_eval} total")

        if len(good) == 1:
            print("\n── Response ─────────────────────────────────────────")
            lines = good[0].text.strip().splitlines()
            print("\n".join(lines[:3]))
            if len(lines) > 3:
                print("…")

    for r in bad:
        print(f"  ERROR: {r.error}")

    print("─" * W)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ollama_bench.py",
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--models",      required=True, help="Ollama model tag")
    p.add_argument("--requests",    type=int, default=1, help="Total requests (default: 1)")
    p.add_argument("--concurrency", type=int, default=1, help="Parallel requests (default: 1)")
    p.add_argument("--chat",        action="store_true", help="Use /api/chat endpoint")
    p.add_argument("--think",       choices=["on", "off"], default="off",
                   help="Enable thinking mode: on | off (default: off)")
    p.add_argument("--context",     type=int, default=104448,
                   help="Context window size in tokens (default: 104448)")
    p.add_argument("--system",      default="", help="System prompt")
    p.add_argument("--prompt",      required=True, help="User prompt")
    p.add_argument("--warmup",      type=int, default=0,
                   help="Throwaway requests before benchmark (default: 0)")
    return p


def main() -> None:
    if len(sys.argv) == 1:
        print(USAGE_EXAMPLE)
        sys.exit(0)

    args = build_parser().parse_args()

    think = args.think == "on"
    mode  = "chat (/api/chat)" if args.chat else "generate (/api/generate)"

    print(f"\n  model       : {args.models}")
    print(f"  mode        : {mode}")
    print(f"  think       : {'on' if think else 'off'}")
    print(f"  context     : {args.context:,}")
    print(f"  requests    : {args.requests}  concurrency: {args.concurrency}")
    print(f"  warmup      : {args.warmup}")
    print(f"  system      : {args.system[:60] or '(none)'}")
    print(f"  prompt      : {args.prompt[:80]}\n")

    asyncio.run(run_warmup(
        model=args.models,
        prompt=args.prompt,
        system=args.system,
        chat=args.chat,
        think=think,
        count=args.warmup,
    ))

    results = asyncio.run(run_batch(
        model=args.models,
        prompt=args.prompt,
        system=args.system,
        chat=args.chat,
        think=think,
        total=args.requests,
        concurrency=args.concurrency,
    ))

    print_table(results)
    print_summary(
        model=args.models,
        results=results,
        think=think,
        context=args.context,
        mode=mode,
        concurrency=args.concurrency,
        warmup=args.warmup,
        system=args.system,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
