"""Typed settings and all default values, loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

  # provider keys (read from .env / environment; never hard-coded)
  openai_api_key: str = ""
  anthropic_api_key: str = ""
  google_api_key: str = ""

  # the single source of every default (behavioural + invocation)
  default_model: str = "openai:gpt-4o-mini"  # provider-prefixed
  default_reasoning_effort: str = ""  # "" = off; low/medium/high for reasoning models
  default_games: int = 10
  default_players: int = 5
  win_rate_window: int = 10  # rolling window for the win-rate trajectory
  memory_window: int = 3  # recent games kept in the memory window

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
