# Terminal-First Profile: `phy-executor`

This document defines the configuration, architecture, and profile characteristics of the repository-local Antigravity custom agent `phy-executor`.

## Files Created / Changed

- `.agents/agents/phy-executor/agent.md`: Workspace-local custom agent definition in Antigravity 1.2.7 Markdown agent format, specifying minimal YAML frontmatter and H1-delimited system instructions.
- `docs/terminal_first_profile.md`: Documentation profile detailing configuration mechanisms, split-machine architecture, and execution commands.

## Split-Machine Architecture

The development and execution lifecycle is split across two dedicated nodes with GitHub acting as the synchronization and handoff boundary:

1. **Local Control Node (Windows Laptop - Intel i5-1240P)**:
   - **Environment**: Host development machine running Antigravity CLI. No local CUDA GPU or RTX 3060 is assumed.
   - **Agent Scope**: The `phy-executor` agent profile is intended to run on the **Local Control Node only**. CKEY is a remote execution target, not an Antigravity development workspace.
   - **Responsibilities**: Code inspection, authoring, refactoring, lightweight dependency-light checks, syntax checks, git diff reviews, commits, and pushes.
   - **Execution Policy**: Heavy Monte Carlo loops, CUDA calibration, GPU benchmarks, and full PHY acceptance test suites are **not** executed locally.

2. **Remote Compute Node (CKEY - Linux with NVIDIA RTX 3060)**:
   - **Environment**: Canonical scientific runtime with project `.venv`, PyTorch with CUDA (`sm_86`), and Sionna 2.0 primitives.
   - **Role & Constraints**: CKEY is an **execution and verification node only, NOT a development/editing node**. Scientific source code must **not** be edited directly on CKEY. If remote verification fails, report the failure and perform the fix on the Local Control Node, then commit/push and rerun remotely.
   - **Responsibilities**: Full PHY acceptance test suites, Monte Carlo calibration runs, GPU benchmarks, and heavy numerical/ML workloads.
   - **Synchronization**: Pulls changes from GitHub (`git pull`).
   - **Verification State**: Scientific changes requiring CUDA/Sionna are considered **pending** until executed and verified on CKEY.

3. **Handoff Workflow**:
   ```text
   [Local Control Node] (phy-executor runs here)
     └── Edit / Refactor Code
     └── Lightweight Static / Logic Check
     └── Inspect git diff
     └── Commit & Push to GitHub
              │
              ▼ (git pull)
   [Remote Compute Node (CKEY)] (Execution / Verification only; no editing)
     └── Run Full PHY Acceptance Tests / PyTorch CUDA Simulations
     └── Generate / Verify Results
              │
              ▼ (if verification fails)
   [Local Control Node]
     └── Investigate failure report -> Apply fix -> Commit & Push -> Rerun on CKEY
   ```

## Exact Custom-Agent Mechanism

- **Platform & Format**: Antigravity 1.2.7 Markdown Agent (`agent.md`) format.
- **Discovery Root**: Project-level `.agents/agents/phy-executor/agent.md`. Discovered automatically by Antigravity's hierarchical workspace walk (`.agents/agents/<agent_name>/`).
- **Configuration Frontmatter**:
  - `name`: `phy-executor`
  - `description`: Terminal-first execution agent for the PHY-ML repository.
  - `mainAgent`: `true` (enables top-level session selection via `--agent phy-executor`)
  - `subagent`: `false`
  - `model`: `inherit` (model and reasoning effort are selected at launch)
  - `commandExecutionPolicy`: `eager`
  - `inheritCustomizations`: `false` (blocks inheritance of ambient global/workspace plugins, rules, and skills)
  - `inheritMcp`: `false` (disables default MCP server auto-mounting)
- **System Instructions**:
  - Enforces direct execution with zero planning ceremony (no unsolicited Implementation Plans).
  - Preserves immutable physics assumptions, BER/BLER semantics, stopping conditions, and ML targets.
  - Guarantees historical result immutability under `results/`.
  - Packages heavy compute as reproducible standalone scripts/commands for execution on CKEY rather than local execution.
  - Enforces local lightweight checks prior to commit and remote CKEY verification for full scientific validation.

## Configuration & Model Selection

- Model selection and reasoning effort are configured dynamically at launch (e.g. `--model gemini-3.8-flash-high --effort high`) rather than hardcoded in the agent frontmatter.
- Custom agent frontmatter uses minimal discovery schema conforming to Antigravity CLI 1.2.7.

## Exact Recommended Launch Command

```bash
agy --agent phy-executor --model gemini-3.8-flash-high --effort high --mode accept-edits --dangerously-skip-permissions
```
