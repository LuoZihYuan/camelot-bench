"""Typed settings loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

  # provider keys (read from .env / environment; never hard-coded)
  openai_api_key: str = ""
  anthropic_api_key: str = ""
  google_api_key: str = ""

  # defaults
  default_model: str = "gpt-4o-mini"  # bare OpenAI model name (Responses API)
  reasoning_effort: str = ""  # "" = off; set low/medium/high for reasoning models
  retries: int = 3

  def export_keys(self) -> None:
    """Push loaded keys into os.environ so Pydantic AI providers find them."""
    for name, value in {
      "OPENAI_API_KEY": self.openai_api_key,
      "ANTHROPIC_API_KEY": self.anthropic_api_key,
      "GOOGLE_API_KEY": self.google_api_key,
    }.items():
      if value and not os.environ.get(name):
        os.environ[name] = value


settings = Settings()
settings.export_keys()
