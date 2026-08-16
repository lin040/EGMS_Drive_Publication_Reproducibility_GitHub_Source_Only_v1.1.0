from __future__ import annotations

import argparse
import gc
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .common import sha256, software_manifest, write_json
from .plots import plot_all
from .reporting import (
    execution_status_table,
    manuscript_results,
    parameter_table,
    study2_tables,
    study3_tables,
    write_table,
)
from .study2 import run_study2
from .study3 import run_study3
from .validation import validate_output


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a mapping")
    return cfg


def _validate_config_contract(cfg: dict[str, Any]) -> None:
    expected_s2 = {
        "Equal weighting": "stress_nll",
        "No alignment": "sas",
        "No temporal consistency": "feature_drift",
        "No modality dropout-distillation": "mean_missing_macro_f1_drop",
    }
    expected_s3 = {
        "No graph": "history_dependent_minFDE_K",
        "Spatial-only graph": "history_dependent_minFDE_K",
        "No predicted-intent gating": "Brier-minFDE_K",
        "Single mode K=1": "MR_1",
    }
    if cfg["study2"].get("primary_endpoints") != expected_s2:
        raise ValueError("Study 2 primary endpoint contract differs from the frozen analysis")
    if cfg["study3"].get("primary_endpoints") != expected_s3:
        raise ValueError("Study 3 primary endpoint contract differs from the frozen analysis")
    if int(cfg["study3"]["modes"]) != 6:
        raise ValueError("This implementation requires K=6 for multi-hypothesis conditions")


def _write_frames(frames: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in frames.items():
        if isinstance(value, pd.DataFrame):
            destination = directory / (
                f"{name}.csv.gz" if name.endswith("predictions") else f"{name}.csv"
            )
            value.to_csv(destination, index=False)


def _write_examples(payload: dict[str, Any], path: Path) -> None:
    import numpy as np

    np.savez_compressed(path, **payload)


def _load_persisted_results(output: Path, prefix: str) -> dict[str, Any]:
    import numpy as np

    results: dict[str, Any] = {}
    for path in sorted((output / "data").glob(f"{prefix}_*.csv*")):
        name = path.name.split(".csv", 1)[0]
        results[name] = pd.read_csv(path)
    if prefix == "study3":
        with np.load(output / "data" / "study3_examples.npz", allow_pickle=True) as payload:
            results["study3_examples"] = {key: payload[key] for key in payload.files}
    return results


def _code_hashes(project_root: Path) -> dict[str, str]:
    candidates = [
        project_root / "run_studies.py",
        project_root / "configs" / "studies23.yaml",
        project_root / "docs" / "STUDIES23_ANALYSIS_PLAN_FROZEN.md",
        project_root / "docs" / "REAL_DATA_EXECUTION_GATE.md",
        project_root / "README.md",
        project_root / "pyproject.toml",
        project_root / "requirements.txt",
        *sorted((project_root / "src" / "egms_studies23").glob("*.py")),
        *sorted((project_root / "tests").glob("*.py")),
    ]
    return {
        str(path.relative_to(project_root)): sha256(path)
        for path in candidates
        if path.is_file()
    }


def _summary_payload(
    cfg: dict[str, Any],
    s2_tables: dict[str, pd.DataFrame],
    s3_tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return json.loads(frame.to_json(orient="records"))

    s2_main = s2_tables["table_s2_main"]
    s3_main = s3_tables["table_s3_main"]
    s2_full = s2_main[
        (s2_main["method"] == "Full")
        & (s2_main["domain"] == "adverse_controlled_injected")
    ]
    s3_full = s3_main[
        (s3_main["method"] == "Full temporal graph") & (s3_main["K"] == 6)
    ]
    return {
        "evidence_label": cfg["project"]["evidence_label"],
        "interpretation_boundary": (
            "Controlled synthetic mechanism validation only; no CARLA, nuScenes, "
            "RADIATE, or Argoverse 2 files were executed."
        ),
        "formal_training_replicate_seeds": cfg["common"]["training_seeds"],
        "study2_full_adverse_selected_metrics": records(
            s2_full[s2_full["metric"].isin(["Action macro-F1", "NLL", "Multiclass Brier (0-2)", "ECE (15 equal-mass bins)"])]
        ),
        "study2_primary_contrasts": records(s2_tables["table_s2_primary_contrasts"]),
        "study3_full_k6_selected_metrics": records(
            s3_full[s3_full["metric"].isin(["Synthetic dominant-maneuver macro-F1", "minADE_6 (m)", "minFDE_6 (m)", "MR_6 at 2 m", "Brier-minFDE_6"])]
        ),
        "study3_primary_contrasts": records(s3_tables["table_s3_primary_contrasts"]),
        "study3_negative_controls": records(s3_tables["table_s3_negative_controls"]),
    }


def _build_analysis_outputs(
    cfg: dict[str, Any],
    output: Path,
    s2_results: dict[str, Any],
    s3_results: dict[str, Any],
) -> None:
    evidence = cfg["project"]["evidence_label"]
    s2_tables = study2_tables(s2_results, cfg)
    s3_tables = study3_tables(s3_results, cfg)
    tables = {
        "table_execution_status": execution_status_table(cfg),
        "table_parameters": parameter_table(cfg),
        **s2_tables,
        **s3_tables,
    }
    for name, frame in tables.items():
        write_table(frame, output / "tables", name)
    plot_all(s2_results, s3_results, s2_tables, s3_tables, output / "figures")
    manuscript = manuscript_results(s2_tables, s3_tables, evidence)
    (output / "MANUSCRIPT_RESULTS_CONTROLLED_SYNTHETIC.md").write_text(
        manuscript, encoding="utf-8"
    )
    write_json(output / "summary.json", _summary_payload(cfg, s2_tables, s3_tables))


def run(config_path: Path, output: Path, overwrite: bool = False) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = Path(__file__).resolve().parents[2]
    cfg = _load_config(config_path)
    _validate_config_contract(cfg)
    initial_code_hashes = _code_hashes(project_root)
    evidence = cfg.get("project", {}).get("evidence_label")
    if evidence != "CONTROLLED_SYNTHETIC_MECHANISM_VALIDATION_NOT_EMPIRICAL_DATASET":
        raise ValueError("The controlled-synthetic evidence label is mandatory")
    if bool(cfg["project"].get("real_datasets_executed")) or bool(cfg["project"].get("carla_executed")):
        raise ValueError("This runner cannot claim CARLA or real-dataset execution")

    allowed_output_root = (project_root / "outputs").resolve()
    resolved_output = output.resolve()
    if resolved_output == allowed_output_root or not resolved_output.is_relative_to(allowed_output_root):
        raise ValueError(f"Output must be a named descendant of {allowed_output_root}")
    if output.is_symlink():
        raise ValueError("Refusing to use a symbolic-link output path")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config_resolved.yaml")
    started = datetime.now(timezone.utc)

    print("[1/5] Running Study 2 controlled synthetic ablations", flush=True)
    s2_results = run_study2(cfg)
    print("[2/5] Running Study 3 controlled synthetic ablations", flush=True)
    s3_results = run_study3(cfg)

    data_dir = output / "data"
    _write_frames(s2_results, data_dir)
    _write_frames(s3_results, data_dir)
    _write_examples(s3_results["study3_examples"], data_dir / "study3_examples.npz")

    print("[3/5] Building manuscript tables and figures", flush=True)
    _build_analysis_outputs(cfg, output, s2_results, s3_results)

    finished = datetime.now(timezone.utc)
    final_code_hashes = _code_hashes(project_root)
    if final_code_hashes != initial_code_hashes:
        raise RuntimeError("Source files changed during execution; refusing to write a provenance manifest")
    manifest = {
        "project_id": cfg["project"]["id"],
        "evidence_label": evidence,
        "carla_executed": False,
        "real_datasets_executed": False,
        "named_dataset_result_rows": 0,
        "pilot_training_seeds": cfg["common"]["pilot_training_seeds"],
        "training_seeds": cfg["common"]["training_seeds"],
        "training_replicate_seeds": cfg["common"]["training_seeds"],
        "training_replicate_definition": (
            "Independent equal-sized training samples from the frozen DGM; all methods "
            "within a replicate share the exact training data; validation/test are fixed."
        ),
        "data_seeds": {
            "study2": cfg["study2"]["data_seed"],
            "study3": cfg["study3"]["data_seed"],
        },
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "config_path": str(config_path.relative_to(project_root)),
        "config_sha256": sha256(config_path),
        "code_sha256": final_code_hashes,
        "software": software_manifest(),
        "analysis_plan_frozen_before_final_full_run": bool(
            cfg["project"]["analysis_plan_frozen_before_final_full_run"]
        ),
        "preregistered": False,
        "ci_scope": (
            "Scene-decomposable primary contrasts: paired crossed bootstrap over training-sample "
            "replicate and test scene. SAS and long summaries: replicate-only t CI conditional "
            "on the fixed retrieval gallery/test scenes."
        ),
        "selection_rule": "No replicate, method, or parameter was selected based on the final full-run result.",
        "validation_rule_note": "No rule requires the Full condition to win.",
    }
    write_json(output / "run_manifest.json", manifest)

    # Release large raw prediction frames before the validator independently
    # reloads and recomputes metrics from the persisted CSV files.
    del s2_results, s3_results
    gc.collect()
    print("[4/5] Running evidence, integrity, metric, and artifact validation", flush=True)
    report = validate_output(output)
    print("[5/5] PASS: formal controlled-synthetic run validated", flush=True)
    return report


def analyze_existing(output: Path) -> dict[str, Any]:
    output = output.resolve()
    project_root = Path(__file__).resolve().parents[2]
    allowed_output_root = (project_root / "outputs").resolve()
    if output == allowed_output_root or not output.is_relative_to(allowed_output_root):
        raise ValueError(f"Output must be a named descendant of {allowed_output_root}")
    config_path = output / "config_resolved.yaml"
    if not config_path.is_file() or not (output / "run_manifest.json").is_file():
        raise FileNotFoundError("Existing output lacks its resolved config or run manifest")
    cfg = _load_config(config_path)
    _validate_config_contract(cfg)
    initial_code_hashes = _code_hashes(project_root)
    print("[1/3] Loading persisted raw predictions and manifests", flush=True)
    s2_results = _load_persisted_results(output, "study2")
    s3_results = _load_persisted_results(output, "study3")
    print("[2/3] Rebuilding tables, manuscript text, and figures", flush=True)
    _build_analysis_outputs(cfg, output, s2_results, s3_results)
    final_code_hashes = _code_hashes(project_root)
    if final_code_hashes != initial_code_hashes:
        raise RuntimeError("Source files changed during analysis; refusing to update provenance")
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "analysis_regenerated_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_revision": (
                "Latency tables use the median across replicate timing summaries with a "
                "percentile-bootstrap interval; "
                "model fits and raw predictions were not rerun or modified."
            ),
            "config_sha256": sha256(config_path),
            "code_sha256": final_code_hashes,
            "software": software_manifest(),
        }
    )
    write_json(output / "run_manifest.json", manifest)
    del s2_results, s3_results
    gc.collect()
    report = validate_output(output)
    print("[3/3] PASS: persisted raw outputs reanalyzed and validated", flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run both controlled simulations")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--overwrite", action="store_true")
    analyze_parser = subparsers.add_parser("analyze", help="Rebuild analysis from persisted raw outputs")
    analyze_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate", help="Validate a completed run")
    validate_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run(args.config, args.output, args.overwrite)
    elif args.command == "analyze":
        analyze_existing(args.output)
    else:
        report = validate_output(args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
