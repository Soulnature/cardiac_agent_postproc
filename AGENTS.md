# Repository Guidelines

## Project Structure & Module Organization
Core logic sits under `cardiac_agent_postproc/`: the orchestrator in `run.py` wires AtlasBuilder, Optimizer, Evaluator, and RevisionController agents, while helpers such as `io_utils.py`, `view_utils.py`, and `ops.py` provide shared utilities. Config files live in `config/` (`default.yaml` for production paths, `sample.yaml` for quick smoke tests). Raw inputs and generated artifacts are staged inside `results/` (SOURCE, TARGET, run subfolders). Root-level scripts (`optimize_labels_no_gt.py`, `train_quality_model.py`, etc.) and the `test_*.py` suite provide entrypoints for manual workflows and regression tests.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create/enter the local environment.
- `pip install -r requirements.txt`: install runtime and tooling dependencies.
- `python -m cardiac_agent_postproc.run --config config/default.yaml`: run the full multi-agent pipeline.
- `python -m cardiac_agent_postproc.run --config config/sample.yaml`: run a minimal dataset for fast iteration.
- `pytest`: execute all tests (IO, integration, vision). Use `pytest test_io.py::TestRoundTrip` for targeted cases.

## Coding Style & Naming Conventions
Use Black-style Python formatting (4-space indent, trailing commas when helpful), snake_case for functions/variables, PascalCase for classes, and explicit type hints on public APIs. Keep configuration keys lowercase with underscores (`paths.source_dir`). Favor deterministic, stateless helpers; pass config dicts rather than relying on module-level globals. Keep comments short and purposeful, explaining intent instead of restating code.

## Testing Guidelines
PyTest discovers every `test_*.py`. Mirror the naming pattern `test_<module>_<behavior>` and cover new agent logic with fixture-backed assertions (e.g., inspect generated CSVs or masks). For substantive changes, run `pytest` plus at least one `python -m cardiac_agent_postproc.run ...` command (default or sample config) to confirm end-to-end integration.

## Commit & Pull Request Guidelines
History follows Conventional-style prefixes (`feat:`, `fix:`, etc.); keep commit subjects ≤72 chars and limit each commit to a cohesive change (code + config + docs as needed). Pull requests should include: concise summary of the change, test evidence (`pytest`, sample run command, or screenshots for plots), and links to tracking issues or experiment notes. Mention any new environment variables or data requirements so reviewers can reproduce agent behavior.

## Agent-Specific Instructions
AtlasBuilder ingests SOURCE predictions/GT to refresh `shape_atlas.pkl`; avoid editing SOURCE contents mid-run. Optimizer runs twice (SOURCE + TARGET) and must never read TARGET GT—respect the `is_target` flag and rely on RQS + edges only. When enabling RevisionController, ensure `.env` exposes the correct LLM endpoint and document chosen guardrail thresholds so other contributors can mirror the setup reliably.
