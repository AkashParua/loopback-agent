"""Model lifecycle + generation on MLX.

Everything that touches MLX runs on ONE dedicated thread (`_EXEC`): the model
is loaded there and every generation runs there. Metal gains nothing from
concurrent requests on a 4B model and they only raise peak memory, so requests
queue behind a single worker instead.

Memory policy (16 GB unified memory is shared with macOS and PyTorch training):
  * lazy load on first request, unload after MLX_KEEP_ALIVE idle seconds
  * mx.set_wired_limit keeps the weights resident (no paging -> stable tok/s)
  * mx.set_memory_limit / set_cache_limit cap what MLX may hold
  * refuse to load when the OS reports less than MLX_MIN_FREE_GB available

Speed:
  * the system prompt of every task is fixed, so its KV/state is computed once
    and snapshotted (prefix cache); a request only prefills its own tokens.
    Qwen 3.5 mixes attention and linear-attention layers whose state cannot be
    trimmed, so we snapshot at the exact system-prompt boundary instead of
    doing longest-common-prefix matching.
  * JSON-schema answers are decoded under an llguidance token mask: no retries,
    and the grammar forces compact JSON (fewer tokens to generate).
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Optional

import numpy as np
import psutil
import structlog

log = structlog.get_logger("mlx_engine")

GB = 1024 ** 3
MODEL_ID = os.getenv("MLX_MODEL", "mlx-community/Qwen3.5-4B-4bit")
KEEP_ALIVE = float(os.getenv("MLX_KEEP_ALIVE", "1800"))  # seconds idle before unload; <=0 never
WIRED_LIMIT_GB = float(os.getenv("MLX_WIRED_LIMIT_GB", "4"))
MEMORY_LIMIT_GB = float(os.getenv("MLX_MEMORY_LIMIT_GB", "6"))
CACHE_LIMIT_GB = float(os.getenv("MLX_CACHE_LIMIT_GB", "0.5"))
MIN_FREE_GB = float(os.getenv("MLX_MIN_FREE_GB", "3"))
MAX_KV = int(os.getenv("MLX_MAX_KV", "8192"))
PREFIX_SLOTS = int(os.getenv("MLX_PREFIX_SLOTS", "16"))

_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


class MemoryPressure(RuntimeError):
    """Not enough free RAM to load the model right now."""


@dataclass
class GenStats:
    ts: float
    kind: str  # "json" | "text"
    prompt_tokens: int
    cached_tokens: int
    gen_tokens: int
    prompt_tps: float
    gen_tps: float
    ttft_s: float
    total_s: float
    peak_gb: float
    finish: str


@dataclass
class EngineState:
    loaded: bool = False
    loading: bool = False
    load_s: Optional[float] = None
    loaded_at: Optional[float] = None
    last_used: Optional[float] = None
    requests: int = 0
    unloads: int = 0
    prefix_hits: int = 0
    prefix_misses: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=200))


def system_memory() -> dict:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {"total_gb": round(vm.total / GB, 2), "available_gb": round(vm.available / GB, 2),
            "used_pct": vm.percent, "swap_used_gb": round(sw.used / GB, 2)}


# ====================================================================== constraint

class JsonConstraint:
    """mlx-lm logits processor that only allows tokens keeping the output valid
    for a JSON schema. Called once per decoding step; tokens[-1] is the token
    sampled at the previous step (the first call sees the prompt tail)."""

    def __init__(self, ll_tokenizer, schema: dict, vocab: int):
        import llguidance
        from llguidance.numpy import allocate_token_bitmask

        grammar = llguidance.LLMatcher.grammar_from_json_schema(
            schema, defaults={"whitespace_flexible": False})
        self.matcher = llguidance.LLMatcher(ll_tokenizer, grammar)
        if self.matcher.is_error():
            raise ValueError(f"bad schema for constrained decoding: {self.matcher.get_error()}")
        self.bitmask = allocate_token_bitmask(1, ll_tokenizer.vocab_size)
        self.vocab = vocab
        self.first = True
        self.failed: Optional[str] = None

    def __call__(self, tokens, logits):
        import mlx.core as mx
        from llguidance.numpy import fill_next_token_bitmask

        if self.failed:
            return logits
        if not self.first:
            self.matcher.consume_token(int(tokens[-1].item()))
        self.first = False
        if self.matcher.is_error():
            self.failed = self.matcher.get_error()
            log.warning("constraint_error", error=self.failed)
            return logits
        fill_next_token_bitmask(self.matcher, self.bitmask)
        allowed = np.unpackbits(self.bitmask.view(np.uint8), bitorder="little")
        mask = np.zeros(self.vocab, dtype=bool)
        n = min(self.vocab, allowed.size)
        mask[:n] = allowed[:n].astype(bool)
        return mx.where(mx.array(mask), logits, mx.array(-float("inf"), dtype=logits.dtype))


# ====================================================================== engine

class Engine:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.ll_tokenizer = None
        self.vocab = 0
        self.im_end: Optional[int] = None
        self.state = EngineState()
        self._prefix: OrderedDict[str, tuple[list[int], list]] = OrderedDict()
        self._lock = threading.Lock()  # guards state fields read by /stats
        self._reaper = threading.Thread(target=self._reap, daemon=True, name="mlx-reaper")
        self._reaper.start()

    # ------------------------------------------------------------------ lifecycle
    def submit(self, fn: Callable, *args, **kwargs):
        return _EXEC.submit(fn, *args, **kwargs)

    def ensure_loaded(self) -> None:
        """Runs on the MLX thread."""
        if self.model is not None:
            return
        import mlx.core as mx
        from llguidance.hf import from_tokenizer
        from mlx_lm import load

        free = psutil.virtual_memory().available / GB
        if free < MIN_FREE_GB:
            raise MemoryPressure(f"only {free:.1f} GB RAM available (< MLX_MIN_FREE_GB={MIN_FREE_GB}); "
                                 "stop a training run or close apps")
        self.state.loading = True
        t0 = time.perf_counter()
        try:
            self._apply_limits()
            model, tokenizer = load(self.model_id)
            hf = tokenizer._tokenizer
            vocab = int(model(mx.array([[0]])).shape[-1])
            eos = [i for i in sorted(tokenizer.eos_token_ids)]
            self.ll_tokenizer = from_tokenizer(hf, n_vocab=vocab, eos_token=eos[0] if eos else None)
            self.model, self.tokenizer, self.vocab = model, tokenizer, vocab
            self.im_end = hf.convert_tokens_to_ids("<|im_end|>")
        finally:
            self.state.loading = False
        with self._lock:
            self.state.loaded, self.state.loaded_at = True, time.time()
            self.state.load_s = round(time.perf_counter() - t0, 2)
            self.state.last_used = time.time()
        log.info("model_loaded", model=self.model_id, secs=self.state.load_s, vocab=vocab,
                 active_gb=round(mx.get_active_memory() / GB, 2))

    def _apply_limits(self) -> None:
        import mlx.core as mx

        info = mx.device_info()
        rec = int(info.get("max_recommended_working_set_size", 0)) or int(MEMORY_LIMIT_GB * GB)
        mx.set_memory_limit(min(int(MEMORY_LIMIT_GB * GB), rec))
        mx.set_cache_limit(int(CACHE_LIMIT_GB * GB))
        try:
            mx.set_wired_limit(min(int(WIRED_LIMIT_GB * GB), rec))
        except (RuntimeError, ValueError) as exc:  # wired limit needs macOS 15+
            log.warning("wired_limit_failed", error=str(exc))

    def unload(self, reason: str = "manual") -> bool:
        """Runs on the MLX thread."""
        if self.model is None:
            return False
        import mlx.core as mx

        self.model = self.tokenizer = self.ll_tokenizer = None
        self._prefix.clear()
        gc.collect()
        mx.clear_cache()
        try:
            mx.set_wired_limit(0)
        except (RuntimeError, ValueError):
            pass
        with self._lock:
            self.state.loaded = False
            self.state.unloads += 1
        log.info("model_unloaded", reason=reason, active_gb=round(mx.get_active_memory() / GB, 2))
        return True

    def _reap(self) -> None:
        while True:
            time.sleep(15)
            st = self.state
            if KEEP_ALIVE > 0 and st.loaded and st.last_used and time.time() - st.last_used > KEEP_ALIVE:
                self.submit(self.unload, f"idle {int(time.time() - st.last_used)}s").result()

    # ------------------------------------------------------------------ prompts
    def render(self, messages: list[dict], template_kwargs: dict) -> tuple[list[int], int, Optional[str]]:
        """Returns (full prompt tokens, length of the cacheable system prefix, prefix key)."""
        kw = {"enable_thinking": False, **template_kwargs}
        tok = self.tokenizer
        full = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kw)
        full = list(full)
        if messages and messages[0]["role"] == "system" and len(messages) > 1:
            # Templates refuse a system-only chat, so render it with a probe user turn and
            # keep the shared tokens up to the end of the system turn (<|im_end|>).
            probe = tok.apply_chat_template([messages[0], {"role": "user", "content": "\x01"}],
                                            add_generation_prompt=False, tokenize=True, **kw)
            n = 0
            while n < min(len(probe), len(full)) and probe[n] == full[n]:
                n += 1
            ends = [i for i in range(n) if full[i] == self.im_end]
            if ends and ends[0] + 1 < len(full):
                cut = ends[0] + 1
                key = hashlib.sha1(json.dumps(full[:cut]).encode()).hexdigest()
                return full, cut, key
        return full, 0, None

    def _prefix_cache(self, key: str, prefix: list[int]) -> list:
        """A private copy of the cache after processing `prefix` (computed once per key)."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        hit = self._prefix.get(key)
        if hit is not None:
            self._prefix.move_to_end(key)
            self.state.prefix_hits += 1
            return copy.deepcopy(hit[1])
        self.state.prefix_misses += 1
        cache = make_prompt_cache(self.model, max_kv_size=MAX_KV)
        arr = mx.array(prefix)
        for i in range(0, len(prefix), 2048):
            self.model(arr[i:i + 2048][None], cache=cache)
            mx.eval([c.state for c in cache])
        self._prefix[key] = (prefix, cache)
        while len(self._prefix) > PREFIX_SLOTS:
            self._prefix.popitem(last=False)
        return copy.deepcopy(cache)

    # ------------------------------------------------------------------ generation
    def generate(self, messages: list[dict], *, max_tokens: int, temperature: float,
                 schema: Optional[dict], template_kwargs: Optional[dict] = None) -> Iterator[dict]:
        """Runs on the MLX thread. Yields {"text": delta} chunks and finally {"done": stats}."""
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        self.ensure_loaded()
        t0 = time.perf_counter()
        tokens, n_prefix, key = self.render(messages, template_kwargs or {})
        if key:
            cache = self._prefix_cache(key, tokens[:n_prefix])
            prompt = tokens[n_prefix:]
        else:
            cache = make_prompt_cache(self.model, max_kv_size=MAX_KV)
            prompt = tokens
        processors = []
        constraint = None
        if schema is not None:
            constraint = JsonConstraint(self.ll_tokenizer, schema, self.vocab)
            processors.append(constraint)
        sampler = make_sampler(temp=temperature, top_p=0.95 if temperature > 0 else 0.0)
        mx.reset_peak_memory()
        ttft = None
        last = None
        n = 0
        for resp in stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens,
                                    sampler=sampler, logits_processors=processors or None,
                                    prompt_cache=cache):
            if ttft is None:
                ttft = time.perf_counter() - t0
            n += 1
            last = resp
            if resp.text:
                yield {"text": resp.text}
        total = time.perf_counter() - t0
        finish = "length" if last is not None and last.finish_reason == "length" else "stop"
        stats = GenStats(
            ts=time.time(), kind="json" if schema is not None else "text",
            prompt_tokens=len(tokens), cached_tokens=n_prefix, gen_tokens=n,
            prompt_tps=round(float(last.prompt_tps), 1) if last else 0.0,
            gen_tps=round(float(last.generation_tps), 1) if last else 0.0,
            ttft_s=round(ttft or total, 3), total_s=round(total, 3),
            peak_gb=round(mx.get_peak_memory() / GB, 2), finish=finish)
        with self._lock:
            self.state.requests += 1
            self.state.last_used = time.time()
            self.state.history.append(stats)
        log.info("generated", **asdict(stats), constraint_error=constraint.failed if constraint else None)
        yield {"done": asdict(stats)}

    # ------------------------------------------------------------------ introspection
    def stats(self) -> dict:
        import mlx.core as mx

        st = self.state
        hist = list(st.history)
        recent = hist[-20:]
        avg = lambda k: round(sum(getattr(h, k) for h in recent) / len(recent), 2) if recent else None  # noqa: E731
        return {
            "model": self.model_id, "loaded": st.loaded, "loading": st.loading, "load_s": st.load_s,
            "loaded_at": st.loaded_at, "last_used": st.last_used,
            "idle_s": round(time.time() - st.last_used, 1) if st.last_used else None,
            "keep_alive_s": KEEP_ALIVE, "requests": st.requests, "unloads": st.unloads,
            "prefix_cache": {"slots": len(self._prefix), "hits": st.prefix_hits, "misses": st.prefix_misses},
            "limits_gb": {"wired": WIRED_LIMIT_GB, "memory": MEMORY_LIMIT_GB, "cache": CACHE_LIMIT_GB,
                          "min_free": MIN_FREE_GB},
            "mlx_memory_gb": {"active": round(mx.get_active_memory() / GB, 2),
                              "cache": round(mx.get_cache_memory() / GB, 2),
                              "peak": round(mx.get_peak_memory() / GB, 2)},
            "system": system_memory(),
            "avg_recent": {"gen_tps": avg("gen_tps"), "prompt_tps": avg("prompt_tps"),
                           "ttft_s": avg("ttft_s"), "total_s": avg("total_s")},
            "history": [asdict(h) for h in hist[-60:]],
        }


engine = Engine()
