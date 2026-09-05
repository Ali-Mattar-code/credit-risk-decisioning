"""Command-line entry point."""

from __future__ import annotations

import argparse
import json

from credit_risk.experiment import run_reproduction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Credit-risk decisioning benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    reproduce = subparsers.add_parser("reproduce", help="Run the full deterministic experiment")
    reproduce.add_argument("--output", default="results/reference")
    reproduce.add_argument("--model", default="artifacts/champion.joblib")
    reproduce.add_argument("--samples", type=int, default=15_000)
    reproduce.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "reproduce":
        summary = run_reproduction(
            output_dir=arguments.output,
            model_path=arguments.model,
            n_applications=arguments.samples,
            seed=arguments.seed,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
