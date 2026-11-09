"""Bounded, approval-first Windows coding-agent runner for synthetic development.

Never merges, deploys, accesses production data, or runs arbitrary tasks from GitHub.
Requires an explicitly approved task file and a locally installed agent executable.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "automation" / "approved_tasks.json"
STATE = ROOT / ".raphael-agent"
BRANCH_PATTERN = re.compile(r"^automation/task-[a-z0-9][a-z0-9-]{0,50}$")
ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{2,60}$")


def run(argv, *, check=True, timeout=60):
    return subprocess.run(argv, cwd=ROOT, text=True, capture_output=True,
                          check=check, timeout=timeout)


def verify_checkout():
    if not (ROOT / ".git").exists():
        raise RuntimeError("Clone the GitHub repository locally before running the agent.")
    status = run(["git", "status", "--porcelain"]).stdout
    if status.strip():
        raise RuntimeError("Working tree is not clean; commit or stash your work first.")
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    if not branch:
        raise RuntimeError("Detached HEAD refused.")
    return branch


def load_task(identifier):
    payload = json.loads(TASKS.read_text(encoding="utf-8"))
    matches = [t for t in payload["tasks"] if t.get("id") == identifier]
    if len(matches) != 1:
        raise RuntimeError("Task must exist exactly once in automation/approved_tasks.json.")
    task = matches[0]
    if not ID_PATTERN.fullmatch(identifier):
        raise RuntimeError("Invalid task identifier.")
    if task.get("approved") is not True or task.get("status") != "ready":
        raise RuntimeError("Task is not explicitly approved and ready.")
    if not isinstance(task.get("objective"), str) or not 20 <= len(task["objective"]) <= 4000:
        raise RuntimeError("Invalid task objective.")
    return task


def build_prompt(task):
    return """You are working in the existing Raphael-AI repository.
Read AGENTS.md, docs/RAPHAEL_MASTER_SPEC.md, docs/REQUIREMENTS_MATRIX.md,
docs/PLATFORM_CONTINUATION.md and the relevant source and tests before editing.
Preserve all tracked requirements and existing code. Use synthetic data only.
No production PHI, external paid APIs, deployment, merge, publishing, or
credential handling. Do not change GitHub workflows or this automation runner.
Do not run git push, git merge, git reset, or destructive commands.
Implement this one approved task with tests, then stop.
Report files changed, tests actually run, and remaining risks.

TASK: """ + task["objective"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Explicitly approved task ID")
    parser.add_argument("--agent", choices=("claude",), default="claude")
    parser.add_argument("--execute", action="store_true",
                        help="Actually invoke Claude Code; default is safe dry run")
    parser.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    if not 60 <= args.max_seconds <= 3600:
        parser.error("max-seconds must be between 60 and 3600.")
    task = load_task(args.task)
    branch = verify_checkout()
    if branch in ("main", "master"):
        raise RuntimeError("Never run coding agent on the default branch.")
    prompt = build_prompt(task)
    print(json.dumps({"task": args.task, "branch": branch,
                      "mode": "execute" if args.execute else "dry-run",
                      "objective": task["objective"]}))
    if not args.execute:
        print("DRY RUN ONLY. No agent was launched.")
        return 0
    if os.environ.get("RAPHAEL_AGENT_APPROVED_TASK") != args.task:
        raise RuntimeError("Set RAPHAEL_AGENT_APPROVED_TASK to this task ID for explicit local approval.")
    executable = shutil.which("claude")
    if not executable:
        raise RuntimeError("Claude Code CLI not found. Install and authenticate it first.")
    STATE.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = STATE / (args.task + "-" + stamp + ".log")
    # Claude's permission controls must be configured by the operator.
    # Never use --dangerously-skip-permissions or bypassPermissions.
    command = [executable, "-p", prompt, "--permission-mode", "default"]
    try:
        with log.open("w", encoding="utf-8") as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output,
                                    stderr=subprocess.STDOUT, timeout=args.max_seconds,
                                    check=False)
        print(json.dumps({"exit_code": result.returncode, "log": str(log),
                          "note": "Review git diff and run tests before any commit or push."}))
        return result.returncode
    except subprocess.TimeoutExpired:
        print("Agent exceeded time budget; inspect worktree and log before retrying.", file=sys.stderr)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
