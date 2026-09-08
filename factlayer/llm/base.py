"""Provider-agnostic LLM interface, caching and JSON handling.

The pipeline never imports a vendor SDK directly. That is what makes the
"runs without credentials" requirement in the brief achievable rather than
aspirational: the same extraction code runs against the Anthropic API, against a
locally installed Claude CLI, or against nothing at all, chosen by one
environment variable.

Two behaviours here matter for a batch job over hundreds of pages:

* **Caching.** Responses are content-addressed by (model, prompt). Re-running the
  pipeline after a code change downstream of extraction costs nothing, and demo
  runs are reproducible.
* **Tolerant JSON parsing.** Models wrap JSON in prose or fences often enough
  that treating it as a hard failure throws away good work. Repair is attempted
  before the call is retried, and a genuine failure is recorded rather than
  silently dropped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar

log = logging.getLogger("factlayer.llm")

T = TypeVar("T")
R = TypeVar("R")


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """Raised when a provider is selected but cannot run (no key, no binary)."""


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


class ResponseCache:
    """Content-addressed on-disk cache. Safe for concurrent use."""

    def __init__(self, directory: str | Path, enabled: bool = True) -> None:
        self.dir = Path(directory)
        self.enabled = enabled
        self._lock = threading.Lock()
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model: str, system: str, user: str) -> str:
        digest = hashlib.sha256()
        digest.update(model.encode())
        digest.update(b"\x00")
        digest.update(system.encode())
        digest.update(b"\x00")
        digest.update(user.encode())
        return digest.hexdigest()

    def get(self, key: str) -> str | None:
        if not self.enabled:
            return None
        path = self.dir / f"{key}.txt"
        if path.exists():
            with self._lock:
                self.hits += 1
            return path.read_text(encoding="utf-8")
        with self._lock:
            self.misses += 1
        return None

    def put(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        tmp = self.dir / f"{key}.tmp"
        tmp.write_text(value, encoding="utf-8")
        tmp.replace(self.dir / f"{key}.txt")


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------


class LLMProvider(ABC):
    name: str = "base"
    supports_batching: bool = True

    def __init__(self, cache: ResponseCache | None = None) -> None:
        self.cache = cache
        self.calls = 0
        self.cached_calls = 0
        self._lock = threading.Lock()

    @abstractmethod
    def _complete(self, system: str, user: str, model: str, max_tokens: int) -> str:
        ...

    @abstractmethod
    def available(self) -> bool:
        ...

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        retries: int = 2,
    ) -> str:
        if self.cache is not None:
            key = self.cache.key(f"{self.name}:{model}", system, user)
            hit = self.cache.get(key)
            if hit is not None:
                with self._lock:
                    self.cached_calls += 1
                return hit
        else:
            key = None

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                out = self._complete(system, user, model, max_tokens)
                with self._lock:
                    self.calls += 1
                if key and self.cache is not None:
                    self.cache.put(key, out)
                return out
            except Exception as exc:  # noqa: BLE001 - provider errors vary widely
                last_error = exc
                if attempt < retries:
                    # Backoff with a little spread so a burst of workers does not
                    # retry in lockstep.
                    time.sleep(1.5 * (2**attempt) + (os.getpid() % 7) * 0.05)
        raise LLMError(f"{self.name} failed after {retries + 1} attempts: {last_error}")

    def complete_json(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        retries: int = 2,
    ) -> Any:
        raw = self.complete(system, user, model, max_tokens=max_tokens, retries=retries)
        parsed = parse_json_loose(raw)
        if parsed is None:
            raise LLMError(f"could not parse JSON from response: {raw[:300]!r}")
        return parsed


# --------------------------------------------------------------------------
# JSON repair
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_loose(text: str) -> Any | None:
    """Best-effort JSON extraction from a model response."""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None

    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = _repair(candidate)
            if repaired is not None:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    continue
    return None


def _json_candidates(text: str) -> Iterable[str]:
    yield text
    for m in _FENCE_RE.finditer(text):
        yield m.group(1).strip()
    # Outermost bracketed span, for responses padded with prose.
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            yield text[start : end + 1]


def _repair(text: str) -> str | None:
    """Fix the failure modes that actually occur: trailing commas, and output
    truncated mid-array by a token limit."""
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    if fixed != text:
        return fixed
    # Truncated array: keep whole elements up to the last balanced one.
    if text.lstrip().startswith("["):
        depth = 0
        in_string = False
        escape = False
        last_complete = -1
        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "[{":
                depth += 1
            elif ch in "]}":
                depth -= 1
                if depth == 1:
                    last_complete = i
        if last_complete > 0:
            return text[: last_complete + 1] + "]"
    return None


# --------------------------------------------------------------------------
# Concurrency helper
# --------------------------------------------------------------------------


def map_concurrent(
    items: Sequence[T],
    fn: Callable[[T], R],
    concurrency: int = 8,
    on_error: Callable[[T, Exception], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[R | None]:
    """Run ``fn`` over ``items`` with bounded concurrency, preserving order.

    Failures are isolated: one bad page does not abort a 300-page document, and
    the caller is handed the exception so it can be recorded as an extraction
    failure rather than vanishing.
    """
    results: list[R | None] = [None] * len(items)
    if not items:
        return results

    done = 0
    lock = threading.Lock()

    def worker(index: int) -> None:
        nonlocal done
        try:
            results[index] = fn(items[index])
        except Exception as exc:  # noqa: BLE001
            if on_error is not None:
                on_error(items[index], exc)
            else:
                log.warning("task %d failed: %s", index, exc)
        finally:
            with lock:
                done += 1
                if progress is not None:
                    progress(done, len(items))

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(worker, range(len(items))))
    return results
