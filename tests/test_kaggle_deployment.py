import ast
import importlib.util
from pathlib import Path

import pytest


def test_deployment_script_pins_package_and_has_discoverable_tasks():
  path = Path(__file__).resolve().parents[1] / "scripts/build_kaggle_task.py"
  spec = importlib.util.spec_from_file_location("build_kaggle_task", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  content = module.build("a" * 40)
  tree = ast.parse(content)
  functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
  assert functions == {"camelot_random_baseline", "camelot_fixed_roster"}
  assert "git+https://github.com/LuoZihYuan/camelot-bench.git@" + "a" * 40 in content
  assert "API_KEY" not in content
  with pytest.raises(ValueError, match="commit SHA"):
    module.build("main")

  invocation = tree.body[-1].value
  assert isinstance(invocation, ast.Call)
  assert invocation.func.value.id == "camelot_random_baseline"
  fixed = ast.parse(module.build("a" * 40, ["opponent"] * 4)).body[-1].value
  assert fixed.func.value.id == "camelot_fixed_roster"
  assert ast.literal_eval(fixed.keywords[1].value) == ["opponent"] * 4
  with pytest.raises(ValueError, match="opponent"):
    module.build("a" * 40, ["opponent"])
