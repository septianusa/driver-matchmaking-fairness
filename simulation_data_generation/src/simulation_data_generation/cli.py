"""Command-line interface for dispatch simulation data generation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from simulation_data_generation.config import load_config
from simulation_data_generation.generator import generate_dataset
from simulation_data_generation.validation import validate_dataset


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def command_generate(args: argparse.Namespace) -> int:
    manifest = generate_dataset(
        args.config,
        scale=args.scale,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"Generated dataset: {manifest['output_data_dir']}")
    print(f"Sample CSV folder: {manifest['sample_dir']}")
    print(f"Road network source: {manifest['road_network_source_type']}")
    print(f"Row counts: {manifest['row_counts']}")
    if not args.no_validate:
        validation_config = load_config(args.config, scale=args.scale)
        result = validate_dataset(
            manifest["output_data_dir"],
            config=validation_config,
            report_dir=Path(manifest["output_data_dir"]).parents[1] / "reports",
        )
        print(f"Validation: {'PASS' if result.ok else 'FAIL'}")
        if result.errors:
            for error in result.errors:
                print(f"  ERROR: {error}")
            return 1
    return 0


def command_validate(args: argparse.Namespace) -> int:
    validation_config = load_config(args.config, scale=args.scale)
    result = validate_dataset(args.data_dir, config=validation_config, report_dir=args.report_dir)
    print(f"Validation: {'PASS' if result.ok else 'FAIL'}")
    print(f"Report: {Path(args.report_dir) / 'validation_summary.md'}")
    if result.errors:
        for error in result.errors:
            print(f"  ERROR: {error}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Surabaya dispatch-matching simulation datasets.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate simulation datasets.")
    generate.add_argument("--config", default="config/default.yaml", help="Path to YAML configuration.")
    generate.add_argument("--scale", default="default", help="Generation scale, for example default, full, or small.")
    generate.add_argument("--output-dir", default=None, help="Optional output directory override.")
    generate.add_argument("--overwrite", action="store_true", help="Replace existing generated data.")
    generate.add_argument("--no-validate", action="store_true", help="Skip validation after generation.")
    generate.set_defaults(func=command_generate)

    validate = subparsers.add_parser("validate", help="Validate generated datasets.")
    validate.add_argument("--config", default="config/default.yaml", help="Path to YAML configuration.")
    validate.add_argument("--scale", default="default", help="Scale to validate against, for example default, full, or small.")
    validate.add_argument("--data-dir", default="data/generated", help="Generated dataset directory.")
    validate.add_argument("--report-dir", default="reports", help="Validation report directory.")
    validate.set_defaults(func=command_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return int(args.func(args))
    except Exception as exc:
        logging.exception("Command failed")
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
