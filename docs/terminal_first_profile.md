# Terminal-First Profile: `phy-executor`

This document defines the configuration, architecture, and profile characteristics of the repository-local Antigravity custom agent `phy-executor`.

## Files Created / Changed

- `.agents/agents/phy-executor/agent.md`: Workspace-local custom agent definition in Antigravity 1.2.7 Markdown agent format, specifying minimal YAML frontmatter and H1-delimited system instructions.
- `docs/terminal_first_profile.md`: Documentation profile detailing configuration mechanisms and execution commands.

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
  - Enforces offloading heavy Monte Carlo and CUDA computation to background Python scripts rather than LLM reasoning loops.
  - Mandates testing before committing and using Git as durable checkpointing.

## Configuration & Model Selection

- Model selection and reasoning effort are configured dynamically at launch (e.g. `--model gemini-3.8-flash-high --effort high`) rather than hardcoded in the agent frontmatter.
- Custom agent frontmatter uses minimal discovery schema conforming to Antigravity CLI 1.2.7.

## Exact Recommended Launch Command

```bash
agy --agent phy-executor --model gemini-3.8-flash-high --effort high --mode accept-edits --dangerously-skip-permissions
```
