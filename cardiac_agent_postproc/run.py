from __future__ import annotations
import argparse
import yaml
from .orchestrator import Orchestrator


def _fallback_summary(verdicts: dict) -> dict:
    summary = {"total": len(verdicts), "good": 0, "approved": 0, "gave_up": 0, "rejected": 0}
    for v in verdicts.values():
        if v in summary:
            summary[v] += 1
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    orch = Orchestrator(cfg)
    results = orch.run_with_revisions()

    # print final overview
    last_key = sorted(results.keys(), key=lambda x: int(x[1:]) if x.startswith("v") else -1)[-1]
    final = results[last_key]
    src_summary = final.get("src_summary") or _fallback_summary(final.get("src_verdicts", {}))
    tgt_summary = final.get("tgt_summary") or _fallback_summary(final.get("tgt_verdicts", {}))
    print("\n=== FINAL SUMMARY ===")
    print(f"Last revision: {last_key}")
    print(f"[SOURCE] Summary: {src_summary}")
    print(f"[TARGET] Summary: {tgt_summary}")
    print(f"[TARGET] Eval:    {final.get('tgt_eval', {})}")

if __name__ == "__main__":
    main()
