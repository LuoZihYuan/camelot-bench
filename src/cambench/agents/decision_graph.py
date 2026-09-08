"""A LangGraph decision subgraph: build -> call -> validate -> (retry | fallback | done)."""

from __future__ import annotations

import os
from typing import Any, Optional, TypedDict

from langgraph.graph import START, END, StateGraph


class DecisionState(TypedDict, total=False):
  system: str
  user: str
  raw: Optional[Any]
  result: Optional[Any]
  ok: bool
  attempts: int
  max_attempts: int


def build_decision_graph(client, schema, validate, fallback):
  """Compile a decision graph bound (via closures) to one schema/validate/fallback.

  Nothing non-serializable (the pydantic schema, the callables) is stored in
  graph state -- they are captured here -- so state stays plain and safe.
  """

  def call(state: DecisionState) -> dict:
    try:
      raw = client.call(state["system"], state["user"], schema)
    except Exception:
      if os.environ.get("CAMBENCH_DEBUG"):
        import traceback

        traceback.print_exc()
      raw = None
    return {"raw": raw, "attempts": state.get("attempts", 0) + 1}

  def validate_node(state: DecisionState) -> dict:
    raw = state.get("raw")
    if raw is None:
      return {"ok": False}
    ok, result = validate(raw)
    return {"ok": ok, "result": result}

  def fallback_node(state: DecisionState) -> dict:
    return {"ok": True, "result": fallback()}

  def route(state: DecisionState) -> str:
    if state.get("ok"):
      return "done"
    if state.get("attempts", 0) < state.get("max_attempts", 2):
      return "retry"
    return "fallback"

  g = StateGraph(DecisionState)
  g.add_node("call", call)
  g.add_node("validate", validate_node)
  g.add_node("fallback", fallback_node)
  g.add_edge(START, "call")
  g.add_edge("call", "validate")
  g.add_conditional_edges("validate", route, {"retry": "call", "fallback": "fallback", "done": END})
  g.add_edge("fallback", END)
  return g.compile()
