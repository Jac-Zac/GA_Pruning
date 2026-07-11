#!/usr/bin/env python3
"""Minimal command line interface for the final GAMO experiment."""

import argparse
import sys

from gamo.run.ablation import run_ablation
from gamo.run.experiment import (
    AblationConfig,
    SearchConfig,
    StructuredConfig,
)
from gamo.run.structured import run_structured
from gamo.run.unstructured import run_unstructured_comparison
from gamo.train.train import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the final GAMO study.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train the FashionMNIST MLP")
    train.add_argument(
        "--force", action="store_true", help="overwrite the existing checkpoint"
    )

    commands.add_parser("structured", help="run the structured comparison")
    commands.add_parser("unstructured", help="run the weight-level comparison")
    commands.add_parser("ablation", help="run the structured-GA sensitivity study")
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.command == "train":
        train_model(force=args.force)
        return

    search = SearchConfig()
    if args.command == "structured":
        run_structured(StructuredConfig(search=search))
    elif args.command == "unstructured":
        run_unstructured_comparison(search)
    else:
        run_ablation(AblationConfig(search=search))


if __name__ == "__main__":
    main()
