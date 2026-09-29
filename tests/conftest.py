"""Exercise the real SDK with an in-memory HTTP transport; never call a provider."""

import json
import os
import re

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["BENCHMARK_CONSOLE_UI"] = "False"
os.environ["BENCHMARK_CONSOLE_QUIET"] = "True"
for name in ("MODEL_PROXY_URL", "MODEL_PROXY_API_KEY", "LLM_DEFAULT", "LLM_DEFAULT_EVAL"):
  os.environ.pop(name, None)

import httpx
import openai
import pytest
import kaggle_benchmarks as kbench
from kaggle_benchmarks.actors.llms import OpenAI


def policy(user, fields):
  match = re.search(r"players: ([^.]+)\.", user)
  names = match.group(1).split(", ") if match else ["A", "B", "C", "D", "E"]
  identity = re.search(r"Your name is (\w+)", user)
  me = identity.group(1) if identity else names[0]
  out = {"reasoning": f"private:{me}"}
  if "team" in fields:
    size = int(re.search(r"Propose a team of (\d+)", user).group(1))
    out["team"] = names[:size]
  elif "statement" in fields:
    choices = user.rsplit("You may nominate: ", 1)[-1].rstrip(".").split(", ")
    out.update(statement=f"public:{me}", next_speaker=choices[0])
  elif "approve" in fields:
    out.update(approve=True, belief=[{"seat": n, "role": "loyal"} for n in names if n != me])
  elif "success" in fields:
    out["success"] = True
  elif "target" in fields:
    out["target"] = names[0]
  elif "reflection" in fields:
    out = {"reflection": f"reflection:{me}"}
  elif "notes" in fields:
    out = {"notes": f"learned:{me}"}
  return out


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path):
  def refuse(*args, **kwargs):
    raise AssertionError("Live HTTP is forbidden in tests")
  monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
  monkeypatch.setattr(kbench.config, "cache_directory", tmp_path / "sdk-cache")
  monkeypatch.setattr(kbench.config, "enable_caching", False)
  monkeypatch.setattr(kbench, "client", kbench.clients.InMemoryClient())


@pytest.fixture
def proxy_factory():
  def make(model="test-candidate", handler=None):
    calls = []
    def dispatch(request):
      payload = json.loads(request.content)
      calls.append(payload)
      if handler is not None:
        custom = handler(payload)
        if custom is not None:
          return custom
      if "response_format" in payload:
        fields = payload["response_format"]["json_schema"]["schema"]["properties"]
      else:
        # SDK 0.6.1 uses a schema prompt for nested models (VoteDecision).
        instruction = payload["messages"][-1]["content"]
        fields = json.loads(instruction[instruction.index("{"):])["properties"]
      user = next(m["content"] for m in payload["messages"] if m["role"] == "user")
      return httpx.Response(200, json={
        "id": "mock", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
          "role": "assistant", "content": json.dumps(policy(user, fields)),
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
      })
    transport = httpx.MockTransport(dispatch)
    client = openai.OpenAI(api_key="test-only", base_url="https://model-proxy.kaggle.com/openapi",
                           http_client=httpx.Client(transport=transport))
    llm = OpenAI(client, model, support_structured_outputs=True, support_temperature=False)
    llm.stream_responses = False
    return llm, calls
  return make
