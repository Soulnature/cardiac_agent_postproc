### API Call Estimate Per Sample

**Current Pipeline Structure (Max 3 Rounds)**:

1. **Round 1**:
   - **Diagnosis Agent**: 1 call (LLM synthesis of rules/VLM).
   - **Planner Agent**: 1 call (LLM planning based on diagnosis).
   - **Executor**: 0 calls (Python logic only).
   - **Verifier**: 1-2 calls (VLM visual comparison + LLM reasoning).
   - **Total**: ~3-4 calls.

2. **Subsequent Rounds (if needed)**:
   - If Round 1 fails/needs work, repeat Diagnosis -> Planner -> Verifier.
   - Total adds up quickly: 3 rounds * 3 agents = **~9-12 calls per sample**.

**Optimization**:
- **Diagnosis**: Identify simple issues (e.g. islands, holes) via rules *without* LLM synthesis?
  - Currently `DiagnosisAgent` uses LLM to synthesize rules and VLM.
  - Optimization: If rules detect ONLY "islands" or "holes" and NO severe issues, skip LLM and return rule-based diagnosis directly.
- **Planner**: Use template plans for simple diagnoses?
  - Optimization: If diagnosis is purely "noise_island", hardcode `remove_islands`.
- **Verifier**: Trust VLM signal more?
  - Currently uses VLM comparison + LLM reasoning.
  - Optimization: If VLM confidence is high (e.g. >0.9), skip LLM reasoning?

**Recommendation**:
For "Worst Cases" (severe misplacement), the full intelligence is needed.
For "Simple Cases", we can bypass LLM.
Since you are batch processing **39 Worst Cases**, keeping full intelligence (3-4 calls/round) is justifiable to ensure quality.
Total for 39 cases: 39 * ~10 = ~400 calls. (Manageable for Gemini Flash).
