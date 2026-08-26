#!/usr/bin/env python3
"""Report how often the classifier agrees with data/labelled_tickets.json.

    python scripts/evaluate.py                       # fake classifier, default seed
    python scripts/evaluate.py --seed 7 --failure-rate 0.3
    CLASSIFIER=anthropic python scripts/evaluate.py  # real model (needs ANTHROPIC_API_KEY)
    python scripts/evaluate.py --min-agreement 0.8   # exit 1 if 'both agree' is below this
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace

from app.config import Settings
from app.evaluate import evaluate, format_report
from app.main import make_classifier


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--failure-rate", type=float)
    ap.add_argument("--min-agreement", type=float, help="fail (exit 1) if 'both agree' is below this fraction")
    args = ap.parse_args()

    settings = Settings.from_env()
    if args.seed is not None:
        settings = replace(settings, fake_seed=args.seed)
    if args.failure_rate is not None:
        settings = replace(settings, fake_failure_rate=args.failure_rate)

    report = asyncio.run(evaluate(make_classifier(settings), max_attempts=settings.max_attempts))
    print(format_report(report))
    if args.min_agreement is not None and report.both_agreement < args.min_agreement:
        print(f"\nFAIL: both-agree {report.both_agreement:.1%} < {args.min_agreement:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
