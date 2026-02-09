# LLM Prompts

This directory contains the system prompts used by the `RevisionControllerAgent`.

- `system_prompt.txt`: The main system prompt used to guide the LLM in proposing parameter revisions.

## How it works

The `RevisionControllerAgent` checks for `system_prompt.txt` in this directory. If found, it uses the content as the system prompt for the LLM. If not found, it falls back to a built-in default prompt.

You can modify `system_prompt.txt` to:
- Inform the LLM about new penalties or strategy changes.
- Adjust the constraints (e.g., "prefer aggressive changes").
- Change the rationale requirements.

## Current Configuration (V5)

The current prompt is tuned for the **V5 Reward Principle**, explicitly listing available penalties like `P_slice_consistency` and `P_touch`, while noting DISABLED penalties like `P_edge_misalignment`.
