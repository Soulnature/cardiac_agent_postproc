#!/usr/bin/env python3
"""Entry-point wrapper that matches your original expectation:
- Phase 0: build/load atlas from SOURCE_DIR GT
- Phase 1: optimize SOURCE (RQS-only)
- Phase 2: optimize TARGET (RQS-only) + Dice/HD95 evaluation
- Optional: revisions loop
"""
from cardiac_agent_postproc.run import main

if __name__ == "__main__":
    main()
