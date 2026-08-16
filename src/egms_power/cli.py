"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_protocol


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the EGMS-Drive prospective protocol power analysis."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate protocol YAML")
    validate.add_argument("--config", default="configs/protocol.yaml")

    run = subparsers.add_parser("run", help="generate all results, tables, and figures")
    run.add_argument("--config", default="configs/protocol.yaml")
    run.add_argument("--output", default="outputs")
    return parser


def _resolve_config(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root() / path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _resolve_config(args.config)
    if args.command == "validate":
        protocol = load_protocol(config)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "version": protocol["project"]["version"],
                    "protocol_sha256": protocol["_meta"]["sha256"],
                },
                indent=2,
            )
        )
        return 0
    from .pipeline import run_power_analysis

    result = run_power_analysis(config, args.output)
    print(json.dumps({"status": "completed", **result["summary"], "validation": result["validation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
