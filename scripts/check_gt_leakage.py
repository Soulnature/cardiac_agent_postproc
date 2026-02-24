#!/usr/bin/env python3
"""
Leakage check utility for strict_no_gt_online runs.

Scans verifier payload artifacts and reports whether online prompt/payload
contains forbidden GT/Oracle tokens.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


FORBIDDEN_PATTERNS = [
    re.compile(r"\bdice\b", re.IGNORECASE),
    re.compile(r"\bhd95\b", re.IGNORECASE),
    re.compile(r"\bground\s*truth\b", re.IGNORECASE),
    re.compile(r"\bgt\b", re.IGNORECASE),
    re.compile(r"\boracle\b", re.IGNORECASE),
]


def _scan_text(text: str) -> List[str]:
    hits: List[str] = []
    for pat in FORBIDDEN_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern)
    return hits


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check strict_no_gt_online leakage in verifier payloads.")
    p.add_argument(
        "--root",
        required=True,
        help="Run output root (will scan recursively for *_verifier_payload.json).",
    )
    p.add_argument(
        "--report_json",
        default=None,
        help="Optional output report JSON path.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Missing root path: {root}")

    payload_files = sorted(root.rglob("*_verifier_payload.json"))
    findings: List[Dict[str, Any]] = []
    checked = 0

    for path in payload_files:
        data = _load_json(path)
        strict_mode = bool(data.get("strict_no_gt_online", False))
        if not strict_mode:
            continue
        checked += 1
        online_prompt = str(data.get("online_prompt", ""))
        online_data = data.get("online_data", {})
        text_blob = online_prompt + "\n" + json.dumps(online_data, ensure_ascii=False)
        hits = _scan_text(text_blob)
        if hits:
            findings.append(
                {
                    "path": str(path),
                    "patterns": hits,
                }
            )

    report = {
        "root": str(root),
        "payload_files_found": int(len(payload_files)),
        "strict_payloads_checked": int(checked),
        "n_findings": int(len(findings)),
        "findings": findings,
    }

    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
