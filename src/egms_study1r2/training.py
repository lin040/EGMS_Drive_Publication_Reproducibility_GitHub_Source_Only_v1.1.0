"""Deterministic paired training for the Study 1-R model comparison.

The caller supplies one method-blind training DataFrame and one external fixed
validation DataFrame.  Both Baseline B and Structured receive identical rows,
labels, epoch count, batch size, learning rate, regularization and deterministic
epoch permutations.  The only intentional differences are their declared
encoders.  The frozen shared head has the same hidden layers for both methods;
their resulting parameter counts are reported rather than silently changing
one architecture after the protocol was frozen.

Each epoch is trained with ``MLPClassifier.partial_fit`` and balanced class
weights.  Model selection uses uncalibrated validation NLL evaluated outside
the estimator after every complete epoch; training never stops early.  The
earliest minimum-NLL epoch is restored, then one temperature is fitted using
the same fixed validation set.  Training history and the reloadable JSON
checkpoint can be persisted beneath a replicate/method output directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .models import (
    MODEL_BASELINE_B,
    MODEL_SLUGS,
    MODEL_STRUCTURED,
    N_CLASSES,
    BaselineBEncoder,
    FeatureSchema,
    StructuredEncoder,
    Study1RClassifier,
    make_classifier,
    mlp_parameter_count,
    multiclass_nll,
)


@dataclass(frozen=True)
class TrainingSpec:
    """Frozen common optimization budget for both Study 1-R methods."""

    epochs: int = 50
    batch_size: int = 128
    learning_rate_init: float = 2.0e-3
    alpha: float = 1.0e-4
    hidden_layer_sizes: tuple[int, ...] = (48, 24)
    class_weighting: str = "inverse_frequency_balanced_from_training_split"
    temperature_lower_bound: float = 0.5
    temperature_upper_bound: float = 5.0
    label_column: str = "y_true"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate_init <= 0.0:
            raise ValueError("learning_rate_init must be positive")
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if not self.hidden_layer_sizes or any(value < 2 for value in self.hidden_layer_sizes):
            raise ValueError("Every hidden-layer width must be at least 2")
        if self.class_weighting != "inverse_frequency_balanced_from_training_split":
            raise ValueError("Study 1-R class weighting must match the frozen protocol")
        if not (0.0 < self.temperature_lower_bound < self.temperature_upper_bound):
            raise ValueError("Temperature bounds must satisfy 0 < lower < upper")
        if not self.label_column:
            raise ValueError("label_column must be non-empty")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "TrainingSpec":
        """Read either a training section or a complete protocol mapping."""

        section: Mapping[str, Any] = config
        temperature: Mapping[str, Any] = {}
        if isinstance(config.get("training"), Mapping):
            section = config["training"]
        elif isinstance(config.get("models"), Mapping):
            models = config["models"]
            if isinstance(models, Mapping):
                shared = models.get("shared_head")
                section = shared if isinstance(shared, Mapping) else models.get("training", models)
                calibrated = models.get("temperature_scaling")
                temperature = calibrated if isinstance(calibrated, Mapping) else {}
        elif isinstance(config.get("shared_head"), Mapping):
            section = config["shared_head"]
            calibrated = config.get("temperature_scaling")
            temperature = calibrated if isinstance(calibrated, Mapping) else {}
        hidden = section.get(
            "hidden_layer_sizes",
            section.get("hidden_layers", (48, 24)),
        )
        if isinstance(hidden, (int, np.integer)):
            hidden = (int(hidden),)
        return cls(
            epochs=int(section.get("epochs", 50)),
            batch_size=int(section.get("batch_size", 128)),
            learning_rate_init=float(
                section.get("learning_rate_init", section.get("learning_rate", 2.0e-3))
            ),
            alpha=float(section.get("alpha", section.get("weight_decay", 1.0e-4))),
            hidden_layer_sizes=tuple(int(value) for value in hidden),
            class_weighting=str(
                section.get(
                    "class_weighting",
                    "inverse_frequency_balanced_from_training_split",
                )
            ),
            temperature_lower_bound=float(temperature.get("lower_bound", 0.5)),
            temperature_upper_bound=float(temperature.get("upper_bound", 5.0)),
            label_column=str(section.get("label_column", "y_true")),
        )


def method_training_specs(
    config: TrainingSpec | Mapping[str, Any],
) -> dict[str, TrainingSpec]:
    """Resolve the explicitly asymmetric Study 1-R2 training budgets.

    Baseline B retains the frozen Study-1R head and optimization budget.  The
    exploratory Structured-R2 candidate may use a separately declared budget;
    this capacity difference is exported in every architecture manifest.
    """

    if isinstance(config, TrainingSpec):
        return {MODEL_BASELINE_B: config, MODEL_STRUCTURED: config}
    models = config.get("models", {})
    if not isinstance(models, Mapping):
        shared = TrainingSpec.from_mapping(config)
        return {MODEL_BASELINE_B: shared, MODEL_STRUCTURED: shared}
    temperature = models.get("temperature_scaling", {})
    shared = models.get("shared_head", {})
    baseline = models.get("baseline_b_training", shared)
    structured = models.get("structured_r2_training", shared)
    if not isinstance(baseline, Mapping) or not isinstance(structured, Mapping):
        raise ValueError("Method-specific training sections must be mappings")

    def resolve(section: Mapping[str, Any]) -> TrainingSpec:
        return TrainingSpec.from_mapping(
            {
                "models": {
                    "shared_head": dict(section),
                    "temperature_scaling": (
                        dict(temperature) if isinstance(temperature, Mapping) else {}
                    ),
                }
            }
        )

    return {
        MODEL_BASELINE_B: resolve(baseline),
        MODEL_STRUCTURED: resolve(structured),
    }


@dataclass
class TrainingResult:
    """One fitted method and its complete epoch-level audit record."""

    model: Study1RClassifier
    history: pd.DataFrame
    best_epoch: int
    best_validation_nll: float
    calibrated_validation_nll: float
    checkpoint_path: Path | None
    checkpoint_sha256: str | None


@dataclass
class PairedTrainingResult:
    """Baseline/Structured results trained under one paired replicate budget."""

    results: dict[str, TrainingResult]
    history: pd.DataFrame
    architecture: pd.DataFrame

    @property
    def models(self) -> dict[str, Study1RClassifier]:
        return {method: result.model for method, result in self.results.items()}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _labels(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"Missing label column: {column}")
    numeric = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.int64)
    if len(numeric) == 0:
        raise ValueError("Training and validation frames must be non-empty")
    if np.any((numeric < 0) | (numeric >= N_CLASSES)):
        raise ValueError("Study 1-R labels must be integers in [0, 3]")
    return numeric


def balanced_class_weights(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sklearn-style balanced class and per-sample weights."""

    y = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(y, minlength=N_CLASSES)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training split is missing required action classes: {missing}")
    class_weights = len(y) / (N_CLASSES * counts.astype(np.float64))
    return class_weights, class_weights[y]


def _partial_fit_supports_weights(model: Study1RClassifier) -> bool:
    return "sample_weight" in inspect.signature(model.estimator.partial_fit).parameters


def _history_row(
    *,
    replicate_id: int,
    method: str,
    epoch: int,
    train_nll: float,
    validation_nll: float,
    n_batches: int,
    training_seed: int,
    spec: TrainingSpec,
    parameter_count: int,
) -> dict[str, Any]:
    return {
        "training_replicate": int(replicate_id),
        "method": method,
        "epoch": int(epoch),
        "train_nll_uncalibrated": float(train_nll),
        "validation_nll_uncalibrated": float(validation_nll),
        "selected_checkpoint": False,
        "epochs_budget": int(spec.epochs),
        "batch_size_budget": int(spec.batch_size),
        "learning_rate_init": float(spec.learning_rate_init),
        "alpha": float(spec.alpha),
        "batches_completed": int(n_batches),
        "training_seed": int(training_seed),
        "parameter_count": int(parameter_count),
    }


def _fit_one(
    *,
    method: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    replicate_id: int,
    training_seed: int,
    spec: TrainingSpec,
    schema: FeatureSchema,
    hidden_layer_sizes: tuple[int, ...],
    output_directory: Path | None,
) -> TrainingResult:
    encoder = (
        BaselineBEncoder(schema)
        if method == MODEL_BASELINE_B
        else StructuredEncoder(schema)
    )
    model = make_classifier(
        method=method,
        encoder=encoder,
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=spec.learning_rate_init,
        alpha=spec.alpha,
        random_state=training_seed,
    )
    x_train_unscaled = encoder.transform(train_frame)
    x_validation_unscaled = encoder.transform(validation_frame)
    model.scaler.fit(x_train_unscaled)
    x_train = model.scaler.transform(x_train_unscaled)
    x_validation = model.scaler.transform(x_validation_unscaled)
    y_train = _labels(train_frame, spec.label_column)
    y_validation = _labels(validation_frame, spec.label_column)
    classes = np.arange(N_CLASSES, dtype=np.int64)
    class_weights, sample_weights = balanced_class_weights(y_train)
    if not _partial_fit_supports_weights(model):
        raise RuntimeError(
            "This Study 1-R implementation requires a scikit-learn release whose "
            "MLPClassifier.partial_fit supports sample_weight (scikit-learn >=1.7)."
        )

    history_rows: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_epoch = -1
    best_validation_nll = float("inf")
    n_rows = len(y_train)
    n_batches = int(np.ceil(n_rows / spec.batch_size))

    for epoch in range(1, spec.epochs + 1):
        # Re-initializing from the same (replicate, epoch) seed yields identical
        # row orders for both methods without sharing mutable RNG state.
        epoch_seed = int(
            np.random.SeedSequence([int(training_seed), int(epoch)]).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        order = np.random.default_rng(epoch_seed).permutation(n_rows)
        for start in range(0, n_rows, spec.batch_size):
            indices = order[start : start + spec.batch_size]
            kwargs: dict[str, Any] = {"sample_weight": sample_weights[indices]}
            if epoch == 1 and start == 0:
                kwargs["classes"] = classes
            model.estimator.partial_fit(x_train[indices], y_train[indices], **kwargs)

        train_probabilities = model.estimator.predict_proba(x_train)
        validation_probabilities = model.estimator.predict_proba(x_validation)
        train_nll = multiclass_nll(y_train, train_probabilities)
        validation_nll = multiclass_nll(y_validation, validation_probabilities)
        history_rows.append(
            _history_row(
                replicate_id=replicate_id,
                method=method,
                epoch=epoch,
                train_nll=train_nll,
                validation_nll=validation_nll,
                n_batches=n_batches,
                training_seed=training_seed,
                spec=spec,
                parameter_count=model.parameter_count(),
            )
        )
        # Strict comparison preserves the earliest epoch on an exact tie.
        if validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            best_epoch = epoch
            model.metadata = {
                "training_replicate": int(replicate_id),
                "training_seed": int(training_seed),
                "best_epoch": int(epoch),
                "best_validation_nll_uncalibrated": float(validation_nll),
                "epochs_budget": int(spec.epochs),
                "batch_size_budget": int(spec.batch_size),
                "learning_rate_init": float(spec.learning_rate_init),
                "alpha": float(spec.alpha),
                "class_counts": np.bincount(y_train, minlength=N_CLASSES).tolist(),
                "class_weights": class_weights.tolist(),
                "train_rows": int(len(train_frame)),
                "validation_rows": int(len(validation_frame)),
            }
            best_payload = model.to_checkpoint_dict()

    if best_payload is None or best_epoch < 1:
        raise RuntimeError("No finite external-validation checkpoint was selected")
    selected = Study1RClassifier.from_checkpoint_dict(best_payload)
    selected.fit_temperature(
        validation_frame,
        y_validation,
        lower_bound=spec.temperature_lower_bound,
        upper_bound=spec.temperature_upper_bound,
    )
    calibrated_validation_nll = multiclass_nll(
        y_validation, selected.predict_proba(validation_frame, calibrated=True)
    )
    selected.metadata.update(
        {
            "temperature": float(selected.temperature),
            "validation_nll_calibrated": float(calibrated_validation_nll),
            "selection_rule": (
                "Earliest minimum uncalibrated NLL on the external fixed validation set; "
                "all configured epochs were trained; temperature fitted after selection."
            ),
        }
    )

    history = pd.DataFrame(history_rows)
    history.loc[history["epoch"].eq(best_epoch), "selected_checkpoint"] = True
    checkpoint_path: Path | None = None
    checkpoint_hash: str | None = None
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = selected.save_checkpoint(output_directory / "best_checkpoint.json")
        checkpoint_hash = sha256(checkpoint_path)
        history.to_csv(output_directory / "training_history.csv", index=False, lineterminator="\n")
        resolved = {
            "schema": "egms-drive-study1r-training-record-1.0",
            "method": method,
            "method_slug": MODEL_SLUGS[method],
            "training_replicate": int(replicate_id),
            "training_seed": int(training_seed),
            "training_spec": asdict(spec),
            "class_counts": np.bincount(y_train, minlength=N_CLASSES).tolist(),
            "class_weights": class_weights.tolist(),
            "encoder": selected.encoder.to_json_dict(),
            "input_features": len(selected.encoder.feature_names),
            "hidden_layer_sizes": list(hidden_layer_sizes),
            "parameter_count": selected.parameter_count(),
            "best_epoch": int(best_epoch),
            "best_validation_nll_uncalibrated": float(best_validation_nll),
            "temperature": float(selected.temperature),
            "validation_nll_calibrated": float(calibrated_validation_nll),
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_hash,
        }
        (output_directory / "resolved_training.json").write_text(
            json.dumps(
                resolved,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    return TrainingResult(
        model=selected,
        history=history,
        best_epoch=best_epoch,
        best_validation_nll=float(best_validation_nll),
        calibrated_validation_nll=float(calibrated_validation_nll),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
    )


def train_paired_models(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    replicate_id: int,
    training_seed: int,
    config: TrainingSpec | Mapping[str, Any],
    output_root: str | Path | None = None,
    schema: FeatureSchema | None = None,
) -> PairedTrainingResult:
    """Train Baseline B and Structured with a common frozen optimization budget.

    Parameters
    ----------
    train_frame, validation_frame:
        Frames containing the observation columns documented in ``models.py``
        plus the configured label column (``y_true`` by default).
    replicate_id, training_seed:
        Recorded formal replicate identifiers.  Both methods receive the same
        training seed and deterministic epoch permutations.
    config:
        ``TrainingSpec`` or a mapping containing ``training``/``models`` keys.
    output_root:
        If provided, artifacts are written under
        ``replicate_XX/{baseline_b|structured}``.
    """

    specs = method_training_specs(config)
    schema = schema or FeatureSchema()
    # Validate labels before either method is trained so both have an identical
    # failure boundary.
    if len({value.label_column for value in specs.values()}) != 1:
        raise ValueError("Both Study 1-R2 methods must use the same label column")
    label_column = specs[MODEL_BASELINE_B].label_column
    _labels(train_frame, label_column)
    _labels(validation_frame, label_column)

    baseline_encoder = BaselineBEncoder(schema)
    structured_encoder = StructuredEncoder(schema)
    # Transforming once here establishes exact dimensions and validates that
    # both representations can be formed from the same source columns.
    baseline_dim = baseline_encoder.transform(train_frame.iloc[:1]).shape[1]
    structured_dim = structured_encoder.transform(train_frame.iloc[:1]).shape[1]
    hidden_layers = {
        method: tuple(specs[method].hidden_layer_sizes)
        for method in (MODEL_BASELINE_B, MODEL_STRUCTURED)
    }
    dimensions = {
        MODEL_BASELINE_B: baseline_dim,
        MODEL_STRUCTURED: structured_dim,
    }

    root = Path(output_root) if output_root is not None else None
    replicate_root = root / f"replicate_{int(replicate_id):02d}" if root else None
    results: dict[str, TrainingResult] = {}
    for method in (MODEL_BASELINE_B, MODEL_STRUCTURED):
        method_output = replicate_root / MODEL_SLUGS[method] if replicate_root else None
        results[method] = _fit_one(
            method=method,
            train_frame=train_frame,
            validation_frame=validation_frame,
            replicate_id=replicate_id,
            training_seed=training_seed,
            spec=specs[method],
            schema=schema,
            hidden_layer_sizes=hidden_layers[method],
            output_directory=method_output,
        )

    history = pd.concat(
        [results[method].history for method in (MODEL_BASELINE_B, MODEL_STRUCTURED)],
        ignore_index=True,
    )
    architecture_rows = []
    target_parameters = mlp_parameter_count(
        baseline_dim, hidden_layers[MODEL_BASELINE_B], N_CLASSES
    )
    for method in (MODEL_BASELINE_B, MODEL_STRUCTURED):
        spec = specs[method]
        count = mlp_parameter_count(dimensions[method], hidden_layers[method], N_CLASSES)
        architecture_rows.append(
            {
                "method": method,
                "input_features": dimensions[method],
                "hidden_layer_sizes": json.dumps(
                    list(hidden_layers[method]), separators=(",", ":")
                ),
                "parameter_count": count,
                "baseline_parameter_target": target_parameters,
                "parameter_difference_from_baseline": count - target_parameters,
                "epochs": spec.epochs,
                "batch_size": spec.batch_size,
                "learning_rate_init": spec.learning_rate_init,
                "alpha": spec.alpha,
            }
        )
    architecture = pd.DataFrame(architecture_rows)
    if replicate_root is not None:
        history.to_csv(
            replicate_root / "training_history.csv", index=False, lineterminator="\n"
        )
        architecture.to_csv(
            replicate_root / "architecture.csv", index=False, lineterminator="\n"
        )

    return PairedTrainingResult(
        results=results,
        history=history,
        architecture=architecture,
    )


def load_paired_checkpoints(
    output_root: str | Path, replicate_id: int
) -> dict[str, Study1RClassifier]:
    """Reload the two deterministic JSON checkpoints for one replicate."""

    replicate_root = Path(output_root) / f"replicate_{int(replicate_id):02d}"
    return {
        method: Study1RClassifier.load_checkpoint(
            replicate_root / MODEL_SLUGS[method] / "best_checkpoint.json"
        )
        for method in (MODEL_BASELINE_B, MODEL_STRUCTURED)
    }


__all__ = [
    "TrainingSpec",
    "TrainingResult",
    "PairedTrainingResult",
    "balanced_class_weights",
    "train_paired_models",
    "load_paired_checkpoints",
    "sha256",
]
