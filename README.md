# ollama-proxy
A lightweight Python proxy that lets any Ollama-compatible app run against [llama.cpp](https://github.com/ggml-org/llama.cpp) directly, without Ollama installed. Perfect as a drop-in replacement, for Ollama while accessing the enhanced performance/features of llama.cpp.

Designed for **MoE models** (like Gemma 4 26B A4B) where you need llama.cpp flags
that Ollama doesn't expose — particularly `--n-cpu-moe` for splitting expert layers
across CPU RAM and GPU VRAM.
