"""Concrete LLM backends.

Three of them, in descending order of what they need from the operator:

``AnthropicProvider``  needs ANTHROPIC_API_KEY. The normal path.
``ClaudeCLIProvider``  needs a locally installed ``claude`` binary and no key.
``NullProvider``       needs nothing and refuses to be called, which forces the
                       pipeline onto its deterministic extractor.

The last one exists because the brief says reviewers must be able to run the
project, and a project that dead-ends without a paid API key does not meet that
bar. Quality drops without a model; the system says so rather than pretending.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

from .base import LLMError, LLMProvider, LLMUnavailable, ResponseCache


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


class NullProvider(LLMProvider):
    """A provider that is never available. Selecting it means 'no LLM'."""

    name = "none"

    def available(self) -> bool:
        return False

    def _complete(self, system: str, user: str, model: str, max_tokens: int) -> str:
        raise LLMUnavailable("no LLM backend configured (FACTLAYER_LLM=none)")


_PROVIDERS = {
    "anthropic": AnthropicProvider,
    "claude_cli": ClaudeCLIProvider,
    "none": NullProvider,
}


def build_provider(name: str, cache: ResponseCache | None = None) -> LLMProvider:
    key = (name or "").strip().lower()
    if key == "auto":
        for candidate in ("anthropic", "claude_cli"):
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
