"""Apple Silicon inference server: Qwen 3.5 4B on MLX behind an OpenAI-compatible API.

Replaces Ollama on macOS. Adds what `mlx_lm.server` lacks and the harness needs:
JSON-schema constrained decoding, per-system-prompt prefix caching, and a
memory manager (lazy load, idle unload, Metal memory caps, single-flight).
"""
