"""Injectable model caller (OpenAI Responses API): (system, user, schema) -> validated output."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

from ..config import settings  # importing this loads .env and re-exports provider keys


class LLMClient:
  """Injected model caller; different players may hold clients for different models."""

  def __init__(
    self,
    model: str | None = None,
    reasoning_effort: str | None = None,
    retries: int | None = None,
  ):
    self.model_name = model or settings.default_model
    # reasoning_effort applies to reasoning models; ignored by others (e.g. gpt-4o-mini)
    self.reasoning_effort = reasoning_effort or settings.reasoning_effort
    self.retries = settings.retries if retries is None else retries

  def call(self, system_prompt: str, user_prompt: str, output_type):
    """One structured call -> validated output_type; game-legality is the caller's job."""
    model = OpenAIResponsesModel(self.model_name)
    kwargs = {}
    if self.reasoning_effort:  # only for reasoning models; omit for gpt-4o-mini etc.
      kwargs["openai_reasoning_effort"] = self.reasoning_effort
    model_settings = OpenAIResponsesModelSettings(**kwargs)
    agent = Agent(
      model,
      output_type=output_type,
      system_prompt=system_prompt,
      model_settings=model_settings,
      retries=self.retries,
    )
    return agent.run_sync(user_prompt).output
