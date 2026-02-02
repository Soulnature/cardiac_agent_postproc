from __future__ import annotations
import argparse, os
import yaml
from .orchestrator import Orchestrator

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
    print("\n=== FINAL SUMMARY ===")
    print(f"Last revision: {last_key}")
    print(f"[SOURCE] RQS: {results[last_key]['src_rqs']}")
    print(f"[TARGET] RQS:  {results[last_key]['tgt_rqs']}")
    print(f"[TARGET] Eval: {results[last_key]['tgt_eval']}")

if __name__ == "__main__":
    main()
