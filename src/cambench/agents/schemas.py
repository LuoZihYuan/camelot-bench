"""Pydantic structured-output schemas for each LLM decision (seats are letters, tags defined in prompt)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..roles import Role


class SeatGuess(BaseModel):
  """A guessed role for one other seat."""

  seat: str = Field(description="Seat letter, e.g. 'C'.")
  role: Role = Field(description="The role you believe this seat holds, from the roles in play.")


class ProposeDecision(BaseModel):
  """Leader's team proposal."""

  reasoning: str = Field(description="(private) Your reasoning for this team.")
  team: list[str] = Field(
    description="(public) Seat letters on the team, e.g. ['A', 'C']. Exactly the "
    "required team size, all distinct. You may include yourself."
  )


class SpeakDecision(BaseModel):
  """One discussion turn: what you say, and who speaks next."""

  reasoning: str = Field(description="(private) Your reasoning.")
  statement: str = Field(description="(public) What you say aloud to the table.")
  next_speaker: str = Field(description="(public) Seat letter of who speaks next. Must be an eligible seat, not yourself.")


class VoteDecision(BaseModel):
  """A vote on the proposed team, with your current read of the table."""

  reasoning: str = Field(description="(private) Your reasoning.")
  belief: list[SeatGuess] = Field(
    description="(private) Your current best guess of the role of each seat you don't already know. One entry per unknown seat."
  )
  approve: bool = Field(description="(public) True to approve the proposed team, False to reject.")


class QuestDecision(BaseModel):
  """A quest card played on a mission."""

  reasoning: str = Field(description="(private) Your reasoning.")
  success: bool = Field(description="(secret) True to make the mission succeed, False to make it fail.")


class AssassinateDecision(BaseModel):
  """The assassin's end-game guess at Merlin."""

  reasoning: str = Field(description="(private) Your reasoning.")
  target: str = Field(description="(public) Seat letter of the player you believe is Merlin.")


class DebriefOutput(BaseModel):
  """Post-game reflection carried into future games."""

  reflection: str = Field(description="(private) Notes to carry into your future games.")
