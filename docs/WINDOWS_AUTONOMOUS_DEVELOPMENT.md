# Raphael Windows coding-agent foundation

This is a **manual, bounded coding-agent runner**, not an always-on autonomous engineer.
It does not schedule itself, automatically approve tasks, merge, push, deploy, run
clinical services, or authorize spending. It is intentionally disabled until
the owner approves a task and explicitly invokes it.

## Windows prerequisites

1. Install Git, Python 3.12, PowerShell and Claude Code CLI, then authenticate
   Claude Code using your own account. Check CLI usage limits and billing.
2. Clone the **existing** Raphael repository, and use a non-default development
   branch based on the latest reviewed Raphael development commit.
3. Read `AGENTS.md` and `docs/PLATFORM_CONTINUATION.md`. Use only synthetic data.
4. In `automation/approved_tasks.json`, change **one** task to
   `"status": "ready"` and `"approved": true` after reviewing its objective.
   Commit that approved task change on your working development branch.
5. Dry run from the repository root:
   `powershell -ExecutionPolicy Bypass -File automation/start-agent.ps1 -Task patient-account-lifecycle`
6. For a bounded real run, in the same PowerShell terminal:
   `$env:RAPHAEL_AGENT_APPROVED_TASK="patient-account-lifecycle"`
   `powershell -ExecutionPolicy Bypass -File automation/start-agent.ps1 -Task patient-account-lifecycle -Execute`
7. Review `git diff`, inspect the local log under `.raphael-agent/`, and
   run applicable Python/frontend/browser tests. Commit and push only after
   human review and successful verification.

The CLI runs with default interactive permissions and a 30-minute execution
limit by default (maximum 60 minutes). It does **not** bypass Claude's
permissions. A task can stop waiting for approval, run out of credits, or fail.
An operator must review any partially edited files before rerunning.

## Security and operational boundaries

- Keep `.raphael-agent/` local; it may contain agent output. Never commit logs,
  credentials, or patient data.
- Use a clean, non-default branch and an isolated synthetic database.
- Do not configure unattended execution until explicit cost limits, secret
  isolation, review gates, and a supervised runner have been validated.
- Agent-generated instructions are not proof of test success or clinical safety.
- Do not grant broad GitHub tokens or production credentials to the agent.
- The existing GitHub Actions checks should verify reviewed commits.

## Next engineering increment

Add tests for runner refusal paths, an isolated workspace strategy,
structured outcome records, and a reviewed task queue. Only then consider
Windows Task Scheduler and supervised recurring execution with explicit
budgets and a kill switch. Never auto-merge or auto-deploy clinical changes.
