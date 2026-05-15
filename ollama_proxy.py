#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------------------
# ollama_proxy : A small drop in replacement for Ollama, that gives you access to the advanced features of llama.cpp
# created for use with VibeBuddy64U but can be used for any Ollama project/uses, specifically designed for enhanced
# performance when using MoE models
#
# https://github.com/brett-olsen/ollama-proxy
# Version 0.2
# Created by Brett Olsen - 2026
# ----------------------------------------------------------------------------------------------------------------------

import asyncio
import base64
import json
import os
import shutil
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

LLAMA_SERVER_BIN = "/usr/local/bin/llama-server"
MODEL_PATH       = "/path/to/your/model.gguf"

# Vision encoder — required for image support.
# Download from the same HF repo as your model:
#   huggingface-cli download unsloth/gemma-4-26B-A4B-it-GGUF \
#     --include "mmproj-BF16.gguf" --local-dir /path/to/models/
# Set to "" or None to disable vision (text-only mode).
MMPROJ_PATH = "/path/to/your/models/mmproj-BF16.gguf"

N_CPU_MOE    = 16   # --n-cpu-moe : MoE expert layers kept on CPU RAM
N_GPU_LAYERS = 99   # -ngl        : transformer layers on GPU (99 = all)

LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 8080
PROXY_PORT = 11434  # keep 11434 — your app needs zero changes

EXTRA_FLAGS: list[str] = [
    "--jinja",
    "--reasoning", "off",       # disable thinking mode (faster)
    "--no-mmap",                # required with --n-cpu-moe
    "--flash-attn", "on",       # hybrid sliding-window attention
    "-ctk",         "q8_0",     # KV cache key quantisation
    "-ctv",         "q8_0",     # KV cache value quantisation
    "-c",           "104448",   # context window
]

# Google's recommended sampling defaults for Gemma 4
DEFAULT_OPTIONS: dict = {
    "temperature": 1.0,
    "top_p":       0.95,
    "top_k":       64,
}

# Set to True to log outgoing params and unload/warmup events
DEBUG: bool = False

# Auto-restart config — if llama-server crashes it will be restarted automatically
RESTART_MAX_RETRIES: int   = 5     # max consecutive restart attempts before giving up
RESTART_DELAY_SECS:  float = 5.0   # seconds to wait between restart attempts


# ══════════════════════════════════════════════════════════════════════════════
#  OLLAMA OPTIONS → llama.cpp / OpenAI PARAMETER MAP
#  Unknown keys pass through unchanged so nothing is silently dropped.
# ══════════════════════════════════════════════════════════════════════════════

OPTION_MAP: dict[str, str] = {
    "num_predict":       "max_tokens",
    "num_ctx":           "n_ctx",
    "temperature":       "temperature",
    "top_p":             "top_p",
    "top_k":             "top_k",
    "repeat_penalty":    "repeat_penalty",
    "presence_penalty":  "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "stop":              "stop",
    "seed":              "seed",
    "tfs_z":             "tfs_z",
    "typical_p":         "typical_p",
    "mirostat":          "mirostat",
    "mirostat_tau":      "mirostat_tau",
    "mirostat_eta":      "mirostat_eta",
    "penalize_newline":  "penalize_nl",
    "num_keep":          "n_keep",
    "repeat_last_n":     "repeat_last_n",
}


# ══════════════════════════════════════════════════════════════════════════════
#  LLAMA-SERVER SUBPROCESS
# ══════════════════════════════════════════════════════════════════════════════

_llama_proc: subprocess.Popen | None = None
_llama_output_lines: list[str] = []
_server_loaded: bool = False           # tracks whether llama-server is up
_server_restarting: bool = False       # True while a restart is in progress
_restart_count: int = 0                # consecutive crash counter
_server_lock: asyncio.Lock | None = None   # prevents concurrent start/stop


def _build_server_cmd() -> list[str]:
    cmd = [
        LLAMA_SERVER_BIN,
        "-m",          MODEL_PATH,
        "-ngl",        str(N_GPU_LAYERS),
        "--n-cpu-moe", str(N_CPU_MOE),
        "--host",      LLAMA_HOST,
        "--port",      str(LLAMA_PORT),
    ]
    if MMPROJ_PATH:
        cmd += ["--mmproj", MMPROJ_PATH]
    cmd += EXTRA_FLAGS
    return cmd


def _pipe_output(stream) -> None:
    """Forward llama-server output to our stdout and keep a rolling buffer."""
    for raw in stream:
        line = raw.rstrip("\n")
        _llama_output_lines.append(line)
        if len(_llama_output_lines) > 200:
            _llama_output_lines.pop(0)
        if DEBUG:
            print(f"[llama-server] {line}", flush=True)
    stream.close()


def _start_llama_server_sync() -> None:
    """Launch llama-server (sync, called from async via run_in_executor)."""
    global _llama_proc, _llama_output_lines
    _llama_output_lines.clear()

    resolved = shutil.which(LLAMA_SERVER_BIN) or (
        LLAMA_SERVER_BIN if os.path.isfile(LLAMA_SERVER_BIN) else None
    )
    if not resolved:
        raise FileNotFoundError(
            f"llama-server not found: {LLAMA_SERVER_BIN!r}\n"
            "Update LLAMA_SERVER_BIN at the top of this script."
        )

    cmd = _build_server_cmd()
    print(f"\n[proxy] Starting llama-server:\n  {' '.join(cmd)}\n", flush=True)

    _llama_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=_pipe_output, args=(_llama_proc.stdout,), daemon=True).start()


def _stop_llama_server_sync() -> None:
    """Terminate llama-server (sync, called from async via run_in_executor)."""
    global _llama_proc
    if _llama_proc and _llama_proc.poll() is None:
        print("[proxy] Stopping llama-server…", flush=True)
        _llama_proc.terminate()
        try:
            _llama_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _llama_proc.kill()
        _llama_proc = None


async def _wait_for_server(timeout: int = 300) -> None:
    """Poll /health until ready. On failure, print captured output."""
    url      = f"http://{LLAMA_HOST}:{LLAMA_PORT}/health"
    deadline = time.monotonic() + timeout

    await asyncio.sleep(3)

    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            if _llama_proc and _llama_proc.poll() is not None:
                tail = "\n".join(_llama_output_lines[-50:])
                raise RuntimeError(
                    f"\n[proxy] llama-server exited (code {_llama_proc.returncode}).\n"
                    f"\n── captured output ──────────────────────────\n{tail}\n"
                    f"─────────────────────────────────────────────\n"
                )
            try:
                r = await client.get(url, timeout=2)
                if r.status_code == 200:
                    print("[proxy] llama-server is ready ✓", flush=True)
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(1)

    raise RuntimeError(
        f"[proxy] llama-server did not become ready within {timeout}s.\n"
        + "\n".join(_llama_output_lines[-20:])
    )


async def _restart_llama_server() -> None:
    """Restart llama-server after an unexpected crash."""
    global _server_loaded, _server_restarting, _restart_count, _client

    async with _server_lock:
        if _server_restarting:
            return   # another coroutine already handling it
        _server_restarting = True
        _server_loaded     = False

    try:
        _restart_count += 1
        print(
            f"\n[proxy] ⚠ llama-server crashed — restart attempt "
            f"{_restart_count}/{RESTART_MAX_RETRIES}…",
            flush=True,
        )

        if _restart_count > RESTART_MAX_RETRIES:
            print(
                f"[proxy] ✗ llama-server has crashed {_restart_count - 1} times. "
                "Giving up. Restart the proxy manually.",
                flush=True,
            )
            return

        # Brief delay so we don't hammer the GPU on repeated crashes
        await asyncio.sleep(RESTART_DELAY_SECS * min(_restart_count, 4))

        # Clean up dead process
        _stop_llama_server_sync()

        # Relaunch
        _start_llama_server_sync()
        await _wait_for_server()

        # Fresh httpx client
        try:
            await _client.aclose()
        except Exception:
            pass
        _client = httpx.AsyncClient(
            base_url=f"http://{LLAMA_HOST}:{LLAMA_PORT}",
            timeout=httpx.Timeout(300.0),
        )

        async with _server_lock:
            _server_loaded     = True
            _server_restarting = False
            _restart_count     = 0   # reset on successful start

        print("[proxy] ✓ llama-server restarted successfully.", flush=True)

    except Exception as e:
        print(f"[proxy] ✗ Restart failed: {e}", flush=True)
        async with _server_lock:
            _server_restarting = False


async def _watchdog() -> None:
    """Background task — polls llama-server every 2s and restarts if it crashed."""
    while True:
        await asyncio.sleep(2)
        if not _server_loaded or _server_restarting:
            continue
        if _llama_proc and _llama_proc.poll() is not None:
            # Process has exited unexpectedly
            print(
                f"[proxy] ⚠ Watchdog detected llama-server exit "
                f"(code {_llama_proc.returncode})",
                flush=True,
            )
            asyncio.create_task(_restart_llama_server())


async def _ensure_server_ready() -> Response | None:
    """Return a 503 response if the server is currently restarting, else None."""
    if _server_restarting:
        return Response(
            content=json.dumps({
                "error":   "llama-server is restarting after a crash, please retry",
                "status":  "loading",
            }),
            status_code=503,
            media_type="application/json",
        )
    if not _server_loaded:
        return Response(
            content=json.dumps({
                "error":  "llama-server is not running",
                "status": "unloaded",
            }),
            status_code=503,
            media_type="application/json",
        )
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  UNLOAD / WARMUP
#  Your app calls these via the standard Ollama API:
#    _ollama_unload → POST /api/generate  {"model": "...", "keep_alive": 0}
#    _ollama_warmup → POST /api/generate  {"model": "...", "keep_alive": -1}
#  The proxy intercepts keep_alive and acts accordingly — zero app changes needed.
# ══════════════════════════════════════════════════════════════════════════════

async def _proxy_unload() -> None:
    """Stop llama-server to free GPU VRAM for other tasks."""
    global _server_loaded
    async with _server_lock:
        if not _server_loaded:
            if DEBUG:
                print("[proxy] unload called but server already stopped", flush=True)
            return
        print("[proxy] UNLOAD — stopping llama-server to free GPU…", flush=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _stop_llama_server_sync)
        _server_loaded = False
        print("[proxy] UNLOAD complete — GPU VRAM freed ✓", flush=True)


async def _proxy_warmup() -> None:
    """Start llama-server if it isn't running and wait for it to be ready."""
    global _server_loaded, _client
    async with _server_lock:
        if _server_loaded:
            if DEBUG:
                print("[proxy] warmup called but server already running", flush=True)
            return
        print("[proxy] WARMUP — loading model into GPU…", flush=True)
        _start_llama_server_sync()
        await _wait_for_server()
        # Re-create the httpx client pointing at the fresh server instance
        try:
            await _client.aclose()
        except Exception:
            pass
        _client = httpx.AsyncClient(
            base_url=f"http://{LLAMA_HOST}:{LLAMA_PORT}",
            timeout=httpx.Timeout(300.0),
        )
        _server_loaded = True
        print("[proxy] WARMUP complete — model ready ✓", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client, _server_loaded, _server_lock
    _server_lock = asyncio.Lock()
    _start_llama_server_sync()
    await _wait_for_server()
    _server_loaded = True
    _client = httpx.AsyncClient(
        base_url=f"http://{LLAMA_HOST}:{LLAMA_PORT}",
        timeout=httpx.Timeout(300.0),
    )
    print(f"[proxy] Listening on http://0.0.0.0:{PROXY_PORT}  (Ollama-compatible)\n")

    watchdog_task = asyncio.create_task(_watchdog())

    yield

    watchdog_task.cancel()
    await _client.aclose()
    _stop_llama_server_sync()


# ══════════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Ollama→llama.cpp proxy", lifespan=lifespan)

_client: httpx.AsyncClient


# ── helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _map_options(options: dict) -> dict:
    merged = {**DEFAULT_OPTIONS, **options}
    mapped = {OPTION_MAP.get(k, k): v for k, v in merged.items()}
    # n_ctx cannot be changed per-request — server startup param only
    mapped.pop("n_ctx", None)
    if DEBUG:
        print(f"[proxy] outgoing params → llama-server: {json.dumps(mapped)}", flush=True)
    return mapped


def _translate_messages(messages: list) -> list:
    """Translate Ollama messages to OpenAI format.

    Ollama image format:
      {"role": "user", "content": "what's in this?", "images": ["base64..."]}

    OpenAI / llama-server format:
      {"role": "user", "content": [
          {"type": "text",      "text": "what's in this?"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]}
    """
    translated = []
    for msg in messages:
        images = msg.get("images")
        if not images:
            # No images — pass through as-is
            translated.append(msg)
            continue

        parts: list = []

        # Text content first (may be string or already a list of parts)
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            parts.extend(content)

        # Append each image as an image_url part
        for img in images:
            # Strip the data URI prefix if the app already added one
            if img.startswith("data:"):
                url = img
            else:
                # Sniff format from base64 header bytes
                mime = _sniff_mime(img)
                url  = f"data:{mime};base64,{img}"

            parts.append({
                "type":      "image_url",
                "image_url": {"url": url},
            })

            if DEBUG:
                print(f"[proxy] image attached — mime={_sniff_mime(img)} "
                      f"size={len(img)} chars", flush=True)

        translated.append({
            "role":    msg.get("role", "user"),
            "content": parts,
        })

    return translated


def _sniff_mime(b64: str) -> str:
    """Detect image MIME type from the first few base64 characters."""
    # Decode just enough bytes to check the magic header
    try:
        header = base64.b64decode(b64[:16] + "==")[:8]
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if header[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if header[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if header[:4] == b"RIFF" or header[:4] == b"WEBP":
            return "image/webp"
    except Exception:
        pass
    return "image/jpeg"   # safe default for photos


def _usage_fields(usage: dict) -> dict:
    return {
        "prompt_eval_count":    usage.get("prompt_tokens", 0),
        "prompt_eval_duration": 0,
        "eval_count":           usage.get("completion_tokens", 0),
        "eval_duration":        0,
        "total_duration":       0,
    }


# ── /api/generate  (intercepts keep_alive for unload/warmup) ─────────────────

@app.post("/api/generate")
async def api_generate(request: Request):
    guard = await _ensure_server_ready()
    if guard:
        return guard

    body    = await request.json()
    model   = body.get("model", "default")
    stream  = body.get("stream", True)
    options = body.get("options", {})

    # ── unload: keep_alive=0 ──────────────────────────────────────────────────
    keep_alive = body.get("keep_alive")
    if keep_alive == 0:
        if DEBUG:
            print("[proxy] intercepted keep_alive=0 → unload", flush=True)
        await _proxy_unload()
        return Response(
            content=json.dumps({"model": model, "done": True}),
            media_type="application/json",
        )

    # ── warmup: keep_alive=-1 (or any negative) ───────────────────────────────
    if isinstance(keep_alive, (int, float)) and keep_alive < 0:
        if DEBUG:
            print(f"[proxy] intercepted keep_alive={keep_alive} → warmup", flush=True)
        await _proxy_warmup()
        return Response(
            content=json.dumps({"model": model, "done": True}),
            media_type="application/json",
        )

    # ── normal generate request ───────────────────────────────────────────────
    prompt  = body.get("prompt", "")
    images  = body.get("images", [])

    # If images are present, convert to a chat-style request so llama-server
    # can handle the multimodal content via /v1/chat/completions
    if images:
        messages = _translate_messages([{
            "role":    "user",
            "content": prompt,
            "images":  images,
        }])
        payload: dict = {
            "model":    model,
            "messages": messages,
            "stream":   stream,
            **_map_options(options),
        }
        if stream:
            return StreamingResponse(_chat_stream(payload, model),
                                     media_type="application/x-ndjson")
        r    = await _client.post("/v1/chat/completions", json=payload)
        data = r.json()
        return {
            "model":      model,
            "created_at": _now(),
            "response":   data["choices"][0]["message"].get("content", ""),
            "done":       True,
            **_usage_fields(data.get("usage", {})),
        }

    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": stream,
        **_map_options(options),
    }

    if stream:
        return StreamingResponse(_gen_stream(payload, model),
                                 media_type="application/x-ndjson")

    r    = await _client.post("/v1/completions", json=payload)
    data = r.json()
    return {
        "model":      model,
        "created_at": _now(),
        "response":   data["choices"][0]["text"],
        "done":       True,
        **_usage_fields(data.get("usage", {})),
    }


async def _gen_stream(payload: dict, model: str) -> AsyncIterator[bytes]:
    async with _client.stream("POST", "/v1/completions", json=payload) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                yield (json.dumps({"model": model, "created_at": _now(),
                                   "response": "", "done": True}) + "\n").encode()
                return
            try:
                chunk  = json.loads(raw)
                choice = chunk["choices"][0]
                yield (json.dumps({
                    "model":      model,
                    "created_at": _now(),
                    "response":   choice.get("text", ""),
                    "done":       choice.get("finish_reason") is not None,
                }) + "\n").encode()
            except (KeyError, json.JSONDecodeError):
                continue


def _sniff_mime(b64: str) -> str:
    """Detect image MIME type from the first bytes of a base64 string."""
    import base64
    try:
        header = base64.b64decode(b64[:16] + "==")[:8]
    except Exception:
        return "image/jpeg"
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if header[:3] == b"GIF":
        return "image/gif"
    if header[:4] == b"RIFF" or header[:4] == b"WEBP":
        return "image/webp"
    return "image/jpeg"   # default — works for JPEG and most others


def _convert_messages(messages: list) -> list:
    """Translate Ollama message format to llama-server OpenAI format.

    Ollama supports two image conventions:
      1. Top-level "images" list on the message  →  ["base64...", ...]
      2. Content parts with type "image"         →  [{"type":"image","data":"base64..."}]

    llama-server expects OpenAI content parts:
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    """
    converted = []
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        images  = msg.get("images", [])   # Ollama top-level images list

        # If content is already a list of parts, normalise image parts in-place
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "image":
                    b64  = part.get("data") or part.get("url", "")
                    mime = _sniff_mime(b64)
                    parts.append({
                        "type":      "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    })
                else:
                    parts.append(part)   # text parts pass through unchanged
            # Append any top-level images that weren't already in content parts
            for b64 in images:
                mime = _sniff_mime(b64)
                parts.append({
                    "type":      "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            converted.append({"role": role, "content": parts})

        elif images:
            # content is a plain string — build a multipart message
            parts: list = []
            if content:
                parts.append({"type": "text", "text": content})
            for b64 in images:
                mime = _sniff_mime(b64)
                parts.append({
                    "type":      "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            converted.append({"role": role, "content": parts})
            if DEBUG:
                print(f"[proxy] converted {len(images)} image(s) in '{role}' message",
                      flush=True)

        else:
            # No images — pass through unchanged
            converted.append(msg)

    return converted


# ── /api/chat ─────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def api_chat(request: Request):
    guard = await _ensure_server_ready()
    if guard:
        return guard

    body     = await request.json()
    model    = body.get("model", "default")
    messages = _translate_messages(body.get("messages", []))
    stream   = body.get("stream", True)
    options  = body.get("options", {})

    payload: dict = {
        "model":    model,
        "messages": _convert_messages(messages),   # handles image translation
        "stream":   stream,
        **_map_options(options),
    }

    if stream:
        return StreamingResponse(_chat_stream(payload, model),
                                 media_type="application/x-ndjson")

    r    = await _client.post("/v1/chat/completions", json=payload)
    data = r.json()
    return {
        "model":      model,
        "created_at": _now(),
        "message":    data["choices"][0]["message"],
        "done":       True,
        **_usage_fields(data.get("usage", {})),
    }


async def _chat_stream(payload: dict, model: str) -> AsyncIterator[bytes]:
    async with _client.stream("POST", "/v1/chat/completions", json=payload) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                yield (json.dumps({
                    "model": model, "created_at": _now(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                }) + "\n").encode()
                return
            try:
                chunk  = json.loads(raw)
                choice = chunk["choices"][0]
                delta  = choice.get("delta", {})
                yield (json.dumps({
                    "model":      model,
                    "created_at": _now(),
                    "message": {
                        "role":    delta.get("role", "assistant"),
                        "content": delta.get("content", ""),
                    },
                    "done": choice.get("finish_reason") is not None,
                }) + "\n").encode()
            except (KeyError, json.JSONDecodeError):
                continue


# ── /api/tags ─────────────────────────────────────────────────────────────────

@app.get("/api/tags")
async def api_tags():
    try:
        r    = await _client.get("/v1/models")
        data = r.json()
        models = [
            {"name": m["id"], "model": m["id"],
             "modified_at": _now(), "size": 0, "details": {}}
            for m in data.get("data", [])
        ]
    except Exception:
        models = [{"name": "local", "model": "local",
                   "modified_at": _now(), "size": 0, "details": {}}]
    return {"models": models}


# ── stubs ─────────────────────────────────────────────────────────────────────

@app.get("/api/version")
async def api_version():
    return {"version": "0.1.0-proxy"}


@app.post("/api/show")
async def api_show(request: Request):
    body = await request.json()
    return {"model": body.get("model", ""), "details": {}, "info": {}}


@app.head("/")
@app.get("/")
async def root():
    return Response(content="Ollama is running", media_type="text/plain")


# ── passthrough ───────────────────────────────────────────────────────────────

@app.api_route("/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"])
async def passthrough(path: str, request: Request):
    body    = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}
    r = await _client.request(
        request.method, f"/{path}",
        content=body, headers=headers,
        params=dict(request.query_params),
    )
    return Response(content=r.content, status_code=r.status_code,
                    headers=dict(r.headers))


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="info")
