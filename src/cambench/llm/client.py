"""Injectable multi-provider model caller (OpenAI / Anthropic / Google) via Pydantic AI."""

from __future__ import annotations

import time

from pydantic_ai import Agent

from ..config import settings  # importing this loads .env and re-exports provider keys

RETRIES = 3  # Pydantic AI retries on schema-invalid output (not user-configurable)
MAX_TRANSIENT_RETRIES = 5  # rate-limit / overload retries with backoff


class LLMClient:
  """Injected model caller; different players may hold clients for different models.

  `model` is a provider-prefixed string (e.g. "openai:gpt-5.6", "anthropic:claude-...",
  "google:gemini-..."). `reasoning_effort` uses Pydantic AI's unified `thinking`
  setting; unsupported levels map to the closest the provider offers, and it is
  silently ignored by non-reasoning models.
  """

  def __init__(self, model: str | None = None, reasoning_effort: str | None = None):
    self.model = model or settings.default_model
    self.reasoning_effort = reasoning_effort if reasoning_effort is not None else settings.default_reasoning_effort

  def call(self, system_prompt: str, user_prompt: str, output_type):
    """One structured call -> validated output_type; game-legality is the caller's job."""
    effort = self.reasoning_effort
    settings_used = {"thinking": effort} if effort else {}
    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
      try:
        return self._run(system_prompt, user_prompt, output_type, settings_used)
      except Exception as e:
        # reasoning-effort rejection: drop thinking and try once more, then continue
        if settings_used and self._is_reasoning_rejection(e):
          self._warn(f"'{self.model}' rejected reasoning_effort='{effort}'; retrying without it.")
          settings_used = {}
          continue
        # transient (rate limit / overload): back off and retry
        if attempt < MAX_TRANSIENT_RETRIES and self._is_transient(e):
          delay = self._retry_delay(e, attempt)
          self._warn(
            f"'{self.model}' transient error ({self._status(e)}); "
            f"retrying in {delay:.0f}s ({attempt + 1}/{MAX_TRANSIENT_RETRIES})."
          )
          time.sleep(delay)
          continue
        raise

  def _run(self, system_prompt, user_prompt, output_type, model_settings):
    agent = Agent(
      self.model,
      output_type=output_type,
      system_prompt=system_prompt,
      model_settings=model_settings,
      retries=RETRIES,
    )
    return agent.run_sync(user_prompt).output

  @staticmethod
  def _warn(msg: str) -> None:
    import sys

    print(f"[cambench] {msg}", file=sys.stderr, flush=True)

  @staticmethod
  def _status(err: Exception):
    return getattr(err, "status_code", "") or ""

  @staticmethod
  def _is_transient(err: Exception) -> bool:
    """Rate-limit / timeout / server errors worth retrying (per Google: 429, 408, 5xx)."""
    code = getattr(err, "status_code", None)
    if code in (408, 429, 500, 502, 503, 504):
      return True
    msg = str(err).lower()
    return any(
      k in msg
      for k in (
        "resource_exhausted",
        "rate limit",
        "overloaded",
        "unavailable",
        "try again",
        "timeout",
        "temporarily",
      )
    )

  @staticmethod
  def _retry_delay(err: Exception, attempt: int) -> float:
    """Server-suggested delay if present, else exponential backoff; both with jitter."""
    import random
    import re

    m = re.search(r"retry in (\d+(?:\.\d+)?)s", str(err), re.IGNORECASE) or re.search(
      r"retrydelay['\"]?:\s*['\"]?(\d+)", str(err), re.IGNORECASE
    )
    base = float(m.group(1)) if m else min(2.0**attempt, 60.0)  # 1, 2, 4, 8, ...
    return min(base + random.uniform(0.0, 1.0), 60.0)  # + jitter

  @staticmethod
  def _is_reasoning_rejection(err: Exception) -> bool:
    """True if the error looks like the model rejecting the reasoning/thinking setting."""
    msg = str(err).lower()
    keys = ("reasoning", "thinking", "reasoning_effort", "reasoning.effort")
    return any(k in msg for k in keys) and (
      "unsupported" in msg or "not supported" in msg or "400" in msg or "invalid" in msg or "responses instead" in msg
    )
