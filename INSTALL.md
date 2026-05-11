# ollama-proxy

A lightweight Python proxy that lets any Ollama-compatible app run against
[llama.cpp](https://github.com/ggml-org/llama.cpp) directly, without Ollama installed.

Designed for **MoE models** (like Gemma 4 26B A4B) where you need llama.cpp flags
that Ollama doesn't expose — particularly `--n-cpu-moe` for splitting expert layers
across CPU RAM and GPU VRAM.

```
Your App  →  POST /api/chat  →  ollama_proxy.py  →  llama-server  →  Model
           (Ollama format)       (translates)       (OpenAI format)
```

## Files

| File | Purpose |
|---|---|
| `ollama_proxy.py` | The proxy — starts llama-server and translates API calls |
| `build_llamacpp.sh` | Builds llama.cpp from source with CUDA support (Arch Linux) |
| `llama_stats.py` | Inspect live llama-server params and slot activity |
| `switch_proxy_onoff.sh` | Toggle between Ollama and ollama-proxy with one command |

---

## Requirements

- Arch Linux (or any systemd distro — adjust package manager as needed)
- NVIDIA GPU with CUDA toolkit installed (`sudo pacman -S cuda`)
- Python 3.11+
- A GGUF model file (see Model section below)

---

## 1. Build llama.cpp

```bash
chmod +x build_llamacpp.sh
./build_llamacpp.sh
```

This clones the latest llama.cpp, compiles with CUDA for your GPU architecture,
and installs binaries to `/usr/local/bin/`.

> **Note:** The script targets `sm86` (RTX 3080 Ti / 3090). Edit
> `DGGML_CUDA_ARCH_LIST` if your GPU differs:
> - RTX 5090 / 5080 → `120`
> - RTX 4090 / 4080 → `89`
> - RTX 3080 Ti / 3090 → `86`
> - RTX 2080 Ti → `75`
> - RTX A100 → `80`

---

## 2. Download a model

The proxy requires a proper GGUF file. **Do not use Ollama's model blobs
directly** — Ollama stores models as internal blobs that llama.cpp cannot
load as standalone files.

### Recommended: Gemma 4 26B A4B (MoE, text + vision)

```bash
pip install huggingface_hub

# UD-Q4_K_M — good balance of quality and size (~17 GB)
huggingface-cli download unsloth/gemma-4-26B-A4B-it-GGUF \
  --include "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" \
  --local-dir /your/models/dir/gemma4-26b/

# Also download the vision encoder (required for image support, ~1.2 GB)
huggingface-cli download unsloth/gemma-4-26B-A4B-it-GGUF \
  --include "mmproj-BF16.gguf" \
  --local-dir /your/models/dir/gemma4-26b/
```

> **Why mmproj?** Gemma 4 is a multimodal model. The vision encoder
> (SigLIP) is stored as a separate GGUF file that llama-server loads via
> `--mmproj`. Without it the server runs in text-only mode and rejects
> image input. Set `MMPROJ_PATH = ""` in the proxy config to intentionally
> disable vision.

### Other quant options

| Quant | Size | Notes |
|---|---|---|
| `UD-Q4_K_M` | ~17 GB | Good default |
| `UD-Q4_K_XL` | ~18 GB | Better quality, Unsloth recommended |
| `Q8_0` | ~28 GB | Near-lossless |
| `UD-Q2_K_XL` | ~11 GB | Smallest, lower quality |

> **⚠ Known issue:** Some unsloth UD quants occasionally emit `<unused49>`
> tokens in long sessions. If you see this, switch to
> `bartowski/google_gemma-4-26B-A4B-it-GGUF` Q4_K_M instead.

### Other models

Any GGUF model works. Popular sources:
- [unsloth](https://huggingface.co/unsloth) — optimised dynamic quants
- [bartowski](https://huggingface.co/bartowski) — wide model selection
- [ggml-org](https://huggingface.co/ggml-org) — official llama.cpp org

---

## 3. Configure the proxy

Edit the configuration block at the top of `ollama_proxy.py`:

```python
# ── Required ──────────────────────────────────────────────────────────────
LLAMA_SERVER_BIN = "/usr/local/bin/llama-server"
MODEL_PATH       = "/path/to/your/model.gguf"

# Vision encoder — required for image support. Set to "" to disable.
MMPROJ_PATH = "/path/to/your/models/mmproj-BF16.gguf"

# ── MoE GPU/CPU split (tune to your hardware) ─────────────────────────────
N_CPU_MOE    = 16   # MoE expert layers kept in CPU RAM (increase if GPU OOMs)
N_GPU_LAYERS = 99   # transformer layers on GPU (99 = all)

# ── Ports ─────────────────────────────────────────────────────────────────
LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 8080    # internal llama-server port
PROXY_PORT = 11434   # Ollama-compatible — your app needs zero changes

# ── Extra llama-server flags ───────────────────────────────────────────────
EXTRA_FLAGS: list[str] = [
    "--jinja",                  # use chat template from GGUF
    "--reasoning", "off",       # disable thinking/chain-of-thought (faster)
    "--no-mmap",                # required with --n-cpu-moe
    "--flash-attn", "on",       # recommended for hybrid attention models
    "-ctk",         "q8_0",     # KV cache key quantisation (saves VRAM)
    "-ctv",         "q8_0",     # KV cache value quantisation
    "-c",           "8192",     # context window
    "-t",           "8",        # CPU threads for MoE expert dispatch
]

# ── Debug ─────────────────────────────────────────────────────────────────
DEBUG: bool = False   # True = log all llama-server output + outgoing params

# ── Sampling defaults (used if your app doesn't override) ─────────────────
DEFAULT_OPTIONS: dict = {
    "temperature": 1.0,
    "top_p":       0.95,
    "top_k":       64,
}
```

### Context window and VRAM

`-c` sets the **total** context split across all slots (default 4 slots).
`-c 8192` gives ~2K per slot. Raise it if your app uses long conversations:

| `-c` value | KV cache RAM (Q8, 4 slots, Gemma 4 26B) |
|---|---|
| `8192` | ~1 GB |
| `32768` | ~4 GB |
| `104448` | ~12 GB |

> **`n_ctx` cannot be changed per-request.** It is a server startup
> parameter. Change `-c` in `EXTRA_FLAGS` and restart the proxy.

---

## 4. Install Python dependencies

```bash
pip install fastapi uvicorn httpx
```

Or in a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn httpx
```

---

## 5. Run

```bash
python ollama_proxy.py
```

The proxy will:
1. Start `llama-server` as a subprocess
2. Wait for the model to load (large models take 30–120s)
3. Open port `11434` — your app connects as if Ollama were running

Test it:
```bash
curl http://localhost:11434/api/tags
curl http://localhost:11434/health
```

---

## 6. Toggle between Ollama and ollama-proxy

`switch_proxy_onoff.sh` lets you swap backends with a single command. Both
services share port `11434` so only one can run at a time.

**First, edit the config at the top of the script:**

```bash
PROXY_USER="your_username"
PROXY_DIR="/path/to/ollama-proxy"
PYTHON_BIN="/usr/bin/python"   # or /path/to/venv/bin/python
```

**Then run it:**

```bash
chmod +x switch_proxy_onoff.sh
sudo ./switch_proxy_onoff.sh
```

What it does on each run:

| Current state | Action |
|---|---|
| ollama-proxy running | Stops proxy → starts Ollama |
| Ollama running | Stops Ollama → starts proxy (creates service if needed) |
| Neither running | Creates service if needed → starts proxy |

The script auto-creates `/etc/systemd/system/ollama-proxy.service` on first
run if it doesn't exist yet.

---

## 7. Run as a systemd service (manual setup)

If you prefer to set up the service manually instead of using the toggle script:

```bash
sudo nano /etc/systemd/system/ollama-proxy.service
```

```ini
[Unit]
Description=Ollama → llama.cpp Proxy
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/path/to/ollama-proxy
ExecStart=/usr/bin/python /path/to/ollama-proxy/ollama_proxy.py
Restart=on-failure
RestartSec=10
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable ollama-proxy
sudo systemctl start ollama-proxy
```

Useful commands:

```bash
sudo systemctl status ollama-proxy
journalctl -u ollama-proxy -f        # live logs
sudo systemctl restart ollama-proxy  # after config changes
sudo systemctl stop ollama-proxy
```

> **venv users:** Point `ExecStart` at the venv Python:
> ```ini
> ExecStart=/path/to/ollama-proxy/venv/bin/python /path/to/ollama-proxy/ollama_proxy.py
> ```

---

## 8. Monitoring with llama_stats.py

```bash
# Server defaults and current slot states
python llama_stats.py

# Live 50ms poll — catches per-request params while a generation is running
python llama_stats.py --live

# Raw JSON from /props (server defaults — never changes per-request)
python llama_stats.py --raw

# Raw JSON from /slots (live state including active params)
python llama_stats.py --slots-raw
```

No extra dependencies — uses Python stdlib only.

> **Note:** `/props` always shows server defaults. Per-request overrides
> (temperature, mirostat, etc.) only appear in `/slots` while a generation
> is active. Use `--live` and send a request to catch them — active params
> are marked with `◀ OVERRIDE`.

---

## Proxy features

### Unload / Warmup (zero app changes needed)

The proxy intercepts the standard Ollama `keep_alive` field:

| Your app sends | Proxy does |
|---|---|
| `POST /api/generate  {"keep_alive": 0}` | Terminates llama-server → frees all VRAM |
| `POST /api/generate  {"keep_alive": -1}` | Starts llama-server if not running, waits for ready |

This lets you free the GPU for other tasks (CUDA apps, games, etc.) and
reload the model on demand — with **no changes to your app code**.

### Image / vision support

The proxy translates Ollama's image format to llama-server's OpenAI content
parts automatically. Your app sends images the normal Ollama way:

```python
{"role": "user", "content": "What's in this image?", "images": ["base64..."]}
```

The proxy converts this to:

```json
{"role": "user", "content": [
    {"type": "text",      "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
]}
```

PNG, JPEG, GIF, and WebP are detected automatically from the image bytes.
Vision requires `MMPROJ_PATH` to be set and the mmproj file downloaded
(see step 2).

### Debug mode

Set `DEBUG = True` in `ollama_proxy.py` to enable:
- All llama-server stdout/stderr forwarded to your terminal
- Outgoing params logged per request
- Unload/warmup events logged

---

## API translation reference

| Ollama endpoint | Translated to |
|---|---|
| `POST /api/chat` | `POST /v1/chat/completions` |
| `POST /api/generate` | `POST /v1/completions` (or `/v1/chat/completions` if images present) |
| `GET /api/tags` | `GET /v1/models` |
| `POST /api/show` | stub response |
| `GET /api/version` | stub response |
| everything else | passed through as-is |

### Options mapping

| Ollama option | llama-server param |
|---|---|
| `num_predict` | `max_tokens` |
| `num_ctx` | *(ignored — startup param only)* |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `top_k` | `top_k` |
| `repeat_penalty` | `repeat_penalty` |
| `repeat_last_n` | `repeat_last_n` |
| `presence_penalty` | `presence_penalty` |
| `frequency_penalty` | `frequency_penalty` |
| `mirostat` | `mirostat` |
| `mirostat_tau` | `mirostat_tau` |
| `mirostat_eta` | `mirostat_eta` |
| `seed` | `seed` |
| `stop` | `stop` |
| `num_keep` | `n_keep` |
| `penalize_newline` | `penalize_nl` |

Unknown options pass through unchanged so nothing is silently dropped.

---

## Troubleshooting

**`wrong number of tensors` on startup**
The model file is an Ollama internal blob — not a standalone GGUF.
Download a proper GGUF from Hugging Face (see step 2).

**`image input is not supported — provide the mmproj`**
Set `MMPROJ_PATH` in the proxy config and download `mmproj-BF16.gguf`
from the same HF repo as your model.

**`cudaMalloc failed: out of memory`**
Reduce `-ngl` to push more transformer layers to CPU RAM, or increase
`N_CPU_MOE` to offload more MoE expert layers.

**`--flash-attn` argument error**
Your llama-server build requires an explicit value. Ensure `EXTRA_FLAGS`
has `"--flash-attn", "on"` as two separate list entries.

**Context not changing between requests**
`num_ctx` / `n_ctx` is a server startup parameter. Change `-c` in
`EXTRA_FLAGS` and restart the proxy.

**Thinking mode is on / responses are slow**
Add `"--reasoning", "off"` to `EXTRA_FLAGS` (or `"--no-thinking"` on
older llama.cpp builds).

**Screen flooded with llama-server messages**
Set `DEBUG = False` (the default) in `ollama_proxy.py`.

**Service fails to start — Python packages not found**
systemd runs in a clean environment. Either install system-wide:
```bash
pip install fastapi uvicorn httpx --break-system-packages
```
Or point `ExecStart` at a venv Python binary.

**`<unused49>` tokens in responses**
Known issue with some unsloth UD quants. Switch to
`bartowski/google_gemma-4-26B-A4B-it-GGUF` Q4_K_M.
