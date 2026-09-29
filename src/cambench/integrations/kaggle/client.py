"""Kaggle-only model calls; no provider credentials or paid-provider fallback."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse

from ...agents.decision_graph import DecisionAborted


@dataclass
class CallBudget:
  """Shared attempt limit across every seat, including failed calls and retries.

  This limits requests, not dollars. Kaggle controls the account's free quota.
  Persist before dispatch so interruptions cannot reset the attempt counter.
  """

  limit: int = 600
  used: int = 0
  persist: Callable[[int], None] | None = None

  def __post_init__(self):
    if type(self.limit) is not int or self.limit < 1 or type(self.used) is not int or self.used < 0:
      raise ValueError("Call limit must be a positive integer and usage non-negative")

  def reserve(self):
    if self.used >= self.limit:
      raise DecisionAborted(f"Kaggle call budget exhausted ({self.used}/{self.limit}); episode is incomplete")
    self.used += 1
    if self.persist:
      try:
        self.persist(self.used)
      except Exception as exc:
        raise DecisionAborted("Cannot persist call budget; refusing further inference") from exc


def require_kaggle_proxy(llm):
  """Reject direct provider clients, even when personal API keys are configured."""
  from kaggle_benchmarks.actors.llms import OpenAI

  if not isinstance(llm, OpenAI):
    raise ValueError("Use a Kaggle ModelProxy with api='openai', not a direct provider client")
  url = urlparse(str(llm.client.base_url))
  host = url.hostname or ""
  if url.scheme != "https" or not any(host == domain or host.endswith("." + domain) for domain in (
    "kaggle.com", "kaggleusercontent.com",
  )):
    raise ValueError("Refusing inference outside the HTTPS Kaggle Model Proxy")


class KaggleClient:
  """Implements LLMPlayer's injected client using a fresh SDK chat per decision."""

  def __init__(self, llm, budget: CallBudget, reasoning_effort: str = "", max_tokens: int = 2048):
    require_kaggle_proxy(llm)
    if reasoning_effort not in ("", "low", "medium", "high"):
      raise ValueError("reasoning_effort must be empty, low, medium, or high")
    if type(max_tokens) is not int or max_tokens < 1:
      raise ValueError("max_tokens must be a positive integer")
    # Do not mutate the evaluated subject, which may be shared by parallel tasks.
    self.llm = copy.copy(llm)
    self.llm.client = llm.client.with_options(max_retries=0, timeout=120.0)
    self.llm.stream_responses = False
    self.budget = budget
    self.model = llm.model
    self.reasoning_effort = reasoning_effort
    self.max_tokens = max_tokens

  def call(self, system_prompt: str, user_prompt: str, output_type):
    import kaggle_benchmarks as kbench

    self.budget.reserve()
    try:
      # These ordinary sub-chats stay in the run's audit trace, but a new chat
      # never inherits its siblings' messages. Only the engine publishes moves.
      with kbench.chats.new(name=output_type.__name__, system_instructions=system_prompt):
        return self.llm.prompt(
          user_prompt,
          schema=output_type,
          reasoning=self.reasoning_effort or None,
          extra_api_params={"max_tokens": self.max_tokens},
        )
    except Exception as exc:
      # Quota/auth/timeouts must not silently become random moves and scores.
      # Do not echo provider response bodies, which may contain private data.
      raise DecisionAborted(f"Kaggle inference failed ({type(exc).__name__}); episode is incomplete") from exc
