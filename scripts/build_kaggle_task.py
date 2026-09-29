"""Build a standalone upload script pinned to a Git commit, without credentials."""

import argparse
from pathlib import Path
import re


def build(source_ref: str, opponent_models: list[str] | None = None) -> str:
  if not re.fullmatch(r"[0-9a-f]{40}", source_ref):
    raise ValueError("source_ref must be a full 40-character Git commit SHA")
  source = Path(__file__).resolve().parents[1] / "benchmarks/kaggle/avalon.py"
  requirement = f"camelot-bench[kaggle] @ git+https://github.com/LuoZihYuan/camelot-bench.git@{source_ref}"
  header = (
    "# Generated deployment file; contains no credentials.\n"
    "import subprocess\nimport sys\n\n"
    "if sys.version_info < (3, 13):\n"
    "    raise RuntimeError('Camelot requires Python 3.13 or later; verify the Kaggle runtime.')\n"
    f"subprocess.check_call([sys.executable, '-m', 'pip', 'install', {requirement!r}])\n\n"
  )
  if opponent_models is None:
    invocation = "camelot_random_baseline.run(llm=kbench.llm)"
  else:
    if not 4 <= len(opponent_models) <= 9 or any(not model.strip() or model == "random" for model in opponent_models):
      raise ValueError("Provide 4–9 explicit Kaggle opponent model slugs")
    invocation = f"camelot_fixed_roster.run(llm=kbench.llm, opponent_models={opponent_models!r})"
  return header + source.read_text() + "\n\n" + invocation + "\n"


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-ref", required=True)
  parser.add_argument("--output", type=Path, default=Path("data/kaggle/avalon_task.py"))
  parser.add_argument("--opponent-models", nargs="+", help="Select the fixed-roster task with 4–9 Kaggle model slugs")
  args = parser.parse_args()
  content = build(args.source_ref, args.opponent_models)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(content)
  print(args.output)
