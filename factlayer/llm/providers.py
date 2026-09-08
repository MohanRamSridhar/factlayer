"""Concrete LLM backends.

Four of them, in descending order of what they need from the operator:

``GeminiProvider``     needs GEMINI_API_KEY, which Google issues free. Default.
``AnthropicProvider``  needs ANTHROPIC_API_KEY. Equivalent quality, paid.
``ClaudeCLIProvider``  needs a locally installed ``claude`` binary and no key.
``NullProvider``       needs nothing and refuses to be called, which forces the
                       pipeline onto its deterministic extractor.

The last one exists because the brief says reviewers must be able to run the
project, and a project that dead-ends without a paid API key does not meet that
bar. Quality drops without a model; the system says so rather than pretending.

Gemini is the default because its free tier makes the project runnable by a
reviewer at no cost. That tier meters *requests*, not tokens, and the newest
models allow very few per day -- which is why the pipeline batches many pages
into each call and why this provider paces itself rather than discovering the
limit through 429s.
"""

from __future__ import annotations

import json
import os
import random
import logging
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

from .base import LLMError, LLMProvider, LLMUnavailable, ResponseCache

log = logging.getLogger("factlayer.llm")


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, cache: ResponseCache | None = None, api_key: str | None = None) -> None:
        super().__init__(cache)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None

    def available(self) -> bool:
        return bool(self.api_key)

    def _client_or_raise(self):
        if self._client is None:
            if not self.api_key:
                raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise LLMUnavailable("the anthropic package is not installed") from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _complete(self, system: str, user: str, model: str, max_tokens: int) -> str:
        client = self._client_or_raise()
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        return "".join(part.text for part in message.content if part.type == "text")


class ClaudeCLIProvider(LLMProvider):
    """Shells out to a local ``claude`` CLI.

    Useful in exactly the situation this project was built in: a developer with
    Claude Code installed and no standalone API key. Slower than the API (each
    call pays process startup) so concurrency matters more here.
    """

    name = "claude_cli"

    def __init__(
        self,
        cache: ResponseCache | None = None,
        binary: str | None = None,
        timeout: int = 240,
    ) -> None:
        super().__init__(cache)
        self.binary = binary or os.environ.get("FACTLAYER_CLAUDE_BIN") or "claude"
        self.timeout = timeout

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def _complete(self, system: str, user: str, model: str, max_tokens: int) -> str:
        if not self.available():
            raise LLMUnavailable(f"{self.binary!r} not found on PATH")
        prompt = f"{system}\n\n---\n\n{user}" if system else user
        proc = subprocess.run(
            [
                self.binary,
                "-p",
                "--model", model,
                "--output-format", "json",
                # No filesystem or network tools: this is a pure text transform,
                # and an extraction job has no business touching the machine.
                "--allowed-tools", "",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise LLMError(f"claude CLI exited {proc.returncode}: {proc.stderr[:300]}")
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise LLMError(f"claude CLI returned non-JSON envelope: {proc.stdout[:200]!r}") from exc
        if payload.get("is_error"):
            raise LLMError(f"claude CLI reported an error: {str(payload)[:300]}")
        return payload.get("result", "") or ""


class _UnsupportedConfig(Exception):
    """The request carried a field this model does not accept."""


class _RateLimiter:
    """Client-side minimum interval between calls, shared across threads.

    The free Gemini tier caps requests per minute. Discovering that cap by
    getting 429s wastes the retry budget on work that was always going to be
    rejected, so the provider simply refuses to issue calls faster than the
    stated rate. ``rpm=0`` disables pacing for paid keys.
    """

    def __init__(self, rpm: int) -> None:
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self.min_interval
        if wait > 0:
            time.sleep(wait)


class GeminiProvider(LLMProvider):
    """Google Gemini via the REST endpoint.

    Deliberately implemented over ``urllib`` rather than the ``google-genai``
    SDK. The call is one POST with a JSON body; taking a dependency for that
    would add an install step to a project whose main promise is that a
    reviewer can run it. It also keeps the failure surface small and readable.

    Two details matter for a batch job on the free tier:

    * ``responseMimeType: application/json`` makes the model emit parseable
      JSON directly, which removes most of the work ``parse_json_loose`` would
      otherwise have to do.
    * Thinking is disabled by default. On 2.5-series models reasoning tokens
      count against the output budget, and extraction is a copying task, not a
      reasoning one -- paying for thinking here buys nothing and halves
      throughput. The adjudicator, which does reason, can re-enable it.
    """

    name = "gemini"
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        cache: ResponseCache | None = None,
        api_key: str | None = None,
        rpm: int | None = None,
        timeout: int = 180,
        thinking_budget: int | None = None,
    ) -> None:
        super().__init__(cache)
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )
        if rpm is None:
            rpm = int(os.environ.get("FACTLAYER_GEMINI_RPM", "10"))
        self.limiter = _RateLimiter(rpm)
        self.timeout = timeout
        if thinking_budget is None:
            thinking_budget = int(os.environ.get("FACTLAYER_GEMINI_THINKING", "0"))
        self.thinking_budget = thinking_budget
        # Models discovered at runtime to reject thinkingConfig.
        self._no_thinking: set[str] = set()

    def available(self) -> bool:
        return bool(self.api_key)

    def _payload(self, system: str, user: str, max_tokens: int, model: str) -> dict:
        config: dict = {
            "temperature": 0.0,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        }
        # Support for thinkingConfig varies by model and the API only says
        # "invalid argument" when it is unwelcome. Rather than hard-code a list
        # that will be wrong next month, the provider tries it once per model
        # and remembers the answer.
        if self.thinking_budget >= 0 and model not in self._no_thinking:
            config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": config,
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def _complete(self, system: str, user: str, model: str, max_tokens: int) -> str:
        if not self.api_key:
            raise LLMUnavailable("GEMINI_API_KEY is not set")

        try:
            return self._post(system, user, model, max_tokens)
        except _UnsupportedConfig:
            # Learn it once, then proceed without the offending field.
            self._no_thinking.add(model)
            log.info("model %s rejects thinkingConfig; retrying without it", model)
            return self._post(system, user, model, max_tokens)

    def _post(self, system: str, user: str, model: str, max_tokens: int) -> str:
        body = json.dumps(self._payload(system, user, max_tokens, model)).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/{model}:generateContent",
            data=body,
            headers={
                "Content-Type": "application/json",
                # Header rather than query string so the key stays out of logs.
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )

        self.limiter.acquire()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if (
                exc.code == 400
                and "INVALID_ARGUMENT" in detail
                and model not in self._no_thinking
                and self.thinking_budget >= 0
            ):
                raise _UnsupportedConfig(detail) from exc
            if exc.code in (429, 500, 503):
                # Rate limited or the model is briefly overloaded. Both resolve
                # by waiting, and both are guaranteed to happen somewhere in a
                # few-hundred-call batch, so sleep past the window rather than
                # burning an immediate retry that will be rejected too.
                time.sleep(20 + random.random() * 10)
                raise LLMError(f"gemini unavailable (HTTP {exc.code}): {detail}") from exc
            if exc.code in (400, 403) and "API_KEY" in detail.upper():
                raise LLMUnavailable(f"gemini rejected the API key: {detail}") from exc
            raise LLMError(f"gemini HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"gemini request failed: {exc.reason}") from exc

        return self._text_from(payload)

    @staticmethod
    def _text_from(payload: dict) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            # Prompt-level block: the whole request was refused, not just a part.
            feedback = payload.get("promptFeedback", {})
            raise LLMError(f"gemini returned no candidates: {str(feedback)[:200]}")

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        if not text.strip():
            reason = candidate.get("finishReason", "unknown")
            if reason == "MAX_TOKENS":
                raise LLMError(
                    "gemini hit maxOutputTokens before emitting text; "
                    "raise max_tokens or lower the thinking budget"
                )
            raise LLMError(f"gemini returned empty text (finishReason={reason})")
        return text


class NullProvider(LLMProvider):
    """A provider that is never available. Selecting it means 'no LLM'."""

    name = "none"

    def available(self) -> bool:
        return False

    def _complete(self, system: str, user: str, model: str, max_tokens: int) -> str:
        raise LLMUnavailable("no LLM backend configured (FACTLAYER_LLM=none)")


_PROVIDERS = {
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
    "claude_cli": ClaudeCLIProvider,
    "none": NullProvider,
}


def build_provider(name: str, cache: ResponseCache | None = None) -> LLMProvider:
    key = (name or "").strip().lower()
    if key == "auto":
        # Free tier first: a reviewer with a Gemini key should not need a paid
        # one, and a developer with both probably meant the free one for a
        # 600-page batch job.
        for candidate in ("gemini", "anthropic", "claude_cli"):
            provider = _PROVIDERS[candidate](cache=cache)
            if provider.available():
                return provider
        return NullProvider(cache=cache)
    cls = _PROVIDERS.get(key)
    if cls is None:
        raise ValueError(
            f"unknown LLM backend {name!r}; expected one of {sorted(_PROVIDERS)} or 'auto'"
        )
    return cls(cache=cache)
