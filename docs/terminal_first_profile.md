# Terminal-First Profile: `phy-executor`

This document defines the configuration, architecture, and profile characteristics of the repository-local Antigravity custom agent `phy-executor`.

## Files Created / Changed

- `.agents/agents/phy-executor/agent.md`: Workspace-local custom agent definition in Antigravity 1.2.7 Markdown agent format, specifying YAML frontmatter and H1-delimited system instructions.
- `docs/terminal_first_profile.md`: Documentation profile detailing configuration mechanisms, skill scoping, and execution commands.

## Exact Custom-Agent Mechanism

- **Platform & Format**: Antigravity 1.2.7 Markdown Agent (`agent.md`) format.
- **Discovery Root**: Project-level `.agents/agents/phy-executor/agent.md`. Discovered automatically by Antigravity's hierarchical workspace walk (`.agents/agents/<agent_name>/`).
- **Configuration Frontmatter**:
  - `name`: `phy-executor`
  - `description`: Workspace-local terminal-first PHY-ML executor optimized for CUDA, PyTorch, Sionna, and direct execution without plan ceremonials.
  - `model`: `gemini-3.8-flash-high`
  - `effort`: `high`
  - `mainAgent`: `true` (enables top-level session selection via `--agent phy-executor`)
  - `inheritCustomizations`: `false` (blocks inheritance of ambient global/workspace plugins, rules, and skills)
  - `inheritMcp`: `false` (disables default MCP server auto-mounting)
  - `skills`: Explicit allowlist restricting active skills strictly to relevant engineering, optimization, and scientific search domains.
- **System Instructions**:
  - Enforces direct execution with zero planning ceremony (no unsolicited Implementation Plans).
  - Preserves immutable physics assumptions, BER/BLER semantics, stopping conditions, and ML targets.
  - Guarantees historical result immutability under `results/`.
  - Enforces offloading heavy Monte Carlo and CUDA computation to background Python scripts rather than LLM reasoning loops.
  - Mandates testing before committing and using Git as durable checkpointing.

## Skill Counts & Scoping

- **Default Discovered Skill Count**: 97 skills (derived from ambient built-in extensions and global plugins under `~/.gemini/config/plugins/`).
- **`phy-executor` Active Skill Count**: 9 skills.

### Retained Skills

1. `agy-customizations`: Comprehensive guide and reference for the Antigravity Customization System.
2. `antigravity-guide`: Quick reference and sitemap for Google Antigravity CLI, IDE, and tools.
3. `google-antigravity-sdk`: Agent architecture, orchestration APIs, and SDK capabilities.
4. `managing-python-dependencies`: Strict Python dependency management avoiding global installs and adhering to local virtualenv/tooling.
5. `uv`: Fast package and environment management workflow utilities.
6. `ml-best-practices`: Structured ML and statistical workflow guidelines for training, evaluation, and comparisons.
7. `literature-search-arxiv`: Automated querying, metadata extraction, and PDF retrieval from arXiv.
8. `literature-search-europepmc`: Biomedical and computational literature search via Europe PMC.
9. `literature-search-openalex`: Bibliometric exploration, citation graphs, and scholarly data access via OpenAlex.

### Excluded Categories

- **Biomedical & Genomics Databases**: ChEMBL, AlphaFold, AlphaGenome, ClinVar, dbSNP, Ensembl, gnomAD, GTEx, Human Protein Atlas, InterPro, JASPAR, NCBI E-utilities, openFDA, OpenTargets, PDB, Predictingthepast, PyMOL, QuickGO, Reactome, STRING, UniBind, UniProt.
- **GCP & Data Engineering**: BigQuery (SQL, BigFrames, AI/ML, Graph, DTS), Bigtable, Cloud Composer / Apache Airflow, Dataflow / Apache Beam, Dataform, Dataproc / Spark, dbt, Cloud Storage (GCS basics, architect, FUSE, security assessment), Lakehouse catalogs, GCP pipeline provisioning.
- **Firebase Application Suite**: Firebase AI Logic, App Hosting, Auth, Basics, Crashlytics, Data Connect, Firestore, Classic Hosting, Remote Config, Security Rules Auditor.
- **Frontend, Browser & Accessibility**: Chrome DevTools, a11y-debugging, Largest Contentful Paint (LCP) optimization, Memory leak debugging, Chrome extension development, Modern web standards.
- **Mobile Development**: Xcode project setup.
- **Real-Time Multimodal Streaming**: Gemini Live API, Gemini Omni Flash API.

## Limitations & Unverified Behavior

1. **Subprocess Nesting Constraints**: Antigravity runtime restrictions and session guidelines prohibit nested execution of `agy` CLI sub-processes from within an active session. Discoverability was verified through binary symbol mapping and path resolution analysis.
2. **Built-in Tool Baseline**: `inheritCustomizations: false` isolates plugins, rules, MCP servers, and ambient skills; core built-in tool primitives (`run_command`, `view_file`, `replace_file_content`, `write_to_file`, `manage_task`) remain accessible by design to support normal development workflows.
3. **Network Sensitivity for Literature Tools**: The retained literature search tools (`arxiv`, `europepmc`, `openalex`) depend on outbound HTTP internet access; in offline or air-gapped environments, these skills will gracefully fail without impacting offline PHY/CUDA workloads.

## Exact Recommended Launch Command

### Interactive Session (Default)
```bash
agy --agent phy-executor --model gemini-3.8-flash-high --effort high
```

### Autonomous Terminal-First Mode (Permission Auto-Approve)
```bash
agy --agent phy-executor --model gemini-3.8-flash-high --effort high --dangerously-skip-permissions
```

### Headless Batch / Single-Prompt Execution
```bash
agy --agent phy-executor --model gemini-3.8-flash-high --effort high --dangerously-skip-permissions -p "<prompt>"
```
