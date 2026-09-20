"""Trainable Study 1-R classifiers and auditable feature encoders.

Both encoders consume exactly the same method-blind observation columns.  For
each modality ``m``, lag ``t`` and semantic variable ``v`` the input frame must
contain::

    {m}_{t}_{v}
    q_{m}_{t}
    available_{m}_{t}

where ``t`` is ``prev`` or ``current``.  The canonical modalities and semantic
variables are imported from :mod:`egms_study1r.generator`; the current frozen
protocol uses ``camera``, ``lidar``, ``radar`` and ``ego`` with ``gap_m``,
``closing_speed_mps``, ``ego_speed_mps``, ``crossing_risk`` and
``route_urgency``.  Identifier, label and ``latent_*`` columns may coexist in
the input DataFrame but are never selected by either encoder.

``BaselineBEncoder`` concatenates quality-gated previous/current modality
measurements and their observed quality/availability indicators.
``StructuredEncoder`` derives reliability-weighted fused means, weighted
disagreements, observation support, temporal changes, quality summaries and
four observation-only physics descriptors from those same columns.  No
method-specific term enters the data generator or either feature schema.

The fitted classifier is a scikit-learn ``MLPClassifier`` trained externally
with ``partial_fit``.  Checkpoints are deterministic, human-auditable JSON;
they contain the scaler, all network arrays, temperature and feature schema,
and can be reloaded without pickle or joblib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:  # The fallback keeps this module importable while release files are built.
    from .common import ACTION_NAMES, MODALITIES, SEMANTIC_VARIABLES
except (ImportError, ModuleNotFoundError):  # pragma: no cover - build-time fallback
    ACTION_NAMES = ("KEEP", "SLOW", "YIELD", "STOP")
    MODALITIES = ("camera", "lidar", "radar", "ego")
    SEMANTIC_VARIABLES = (
        "gap_m",
        "closing_speed_mps",
        "ego_speed_mps",
        "crossing_risk",
        "route_urgency",
    )


N_CLASSES = len(ACTION_NAMES)
MODEL_BASELINE_B = "Baseline B"
MODEL_STRUCTURED = "Structured-R2"
MODEL_SLUGS = {
    MODEL_BASELINE_B: "baseline_b",
    MODEL_STRUCTURED: "structured_r2",
}
EPS = 1.0e-12


def _finite_float_list(values: np.ndarray) -> list[Any]:
    """Return nested JSON floats while refusing NaN/Inf checkpoint values."""

    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("Checkpoint array contains a non-finite value")
    return array.tolist()


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != N_CLASSES:
        raise ValueError(f"Expected logits with shape [N,{N_CLASSES}], found {values.shape}")
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), EPS)


def multiclass_nll(labels: Iterable[int], probabilities: np.ndarray) -> float:
    """Mean multiclass negative log likelihood with strict shape checks."""

    y = np.asarray(list(labels), dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(y), N_CLASSES):
        raise ValueError(f"Probability shape {p.shape} is incompatible with {len(y)} labels")
    if len(y) == 0 or np.any((y < 0) | (y >= N_CLASSES)):
        raise ValueError("Labels must be non-empty integers in [0, 3]")
    if not np.all(np.isfinite(p)) or np.any(p < -1.0e-12):
        raise ValueError("Probabilities must be finite and non-negative")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1.0e-8, rtol=0.0):
        raise ValueError("Probability rows do not sum to one")
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], EPS)).mean())


@dataclass(frozen=True)
class FeatureSchema:
    """Ordered method-blind observation schema stored in every checkpoint."""

    modalities: tuple[str, ...] = tuple(MODALITIES)
    semantic_variables: tuple[str, ...] = tuple(SEMANTIC_VARIABLES)
    lags: tuple[str, ...] = ("prev", "current")

    def __post_init__(self) -> None:
        if not self.modalities or not self.semantic_variables:
            raise ValueError("Feature schema requires modalities and semantic variables")
        if self.lags != ("prev", "current"):
            raise ValueError("Study 1-R requires ordered lags ('prev', 'current')")
        for value in (*self.modalities, *self.semantic_variables):
            if not value or not isinstance(value, str):
                raise ValueError("Feature-schema names must be non-empty strings")

    @property
    def observation_columns(self) -> tuple[str, ...]:
        columns: list[str] = []
        for lag in self.lags:
            for modality in self.modalities:
                columns.extend(
                    f"{modality}_{lag}_{variable}"
                    for variable in self.semantic_variables
                )
                columns.append(f"q_{modality}_{lag}")
                columns.append(f"available_{modality}_{lag}")
        return tuple(columns)

    def to_json_dict(self) -> dict[str, list[str]]:
        return {
            "modalities": list(self.modalities),
            "semantic_variables": list(self.semantic_variables),
            "lags": list(self.lags),
        }

    @classmethod
    def from_json_dict(cls, value: Mapping[str, Sequence[str]]) -> "FeatureSchema":
        return cls(
            modalities=tuple(value["modalities"]),
            semantic_variables=tuple(value["semantic_variables"]),
            lags=tuple(value["lags"]),
        )


class ObservationEncoder:
    """Base class for deterministic, stateless Study 1-R encoders."""

    kind = "abstract"

    def __init__(self, schema: FeatureSchema | None = None) -> None:
        self.schema = schema or FeatureSchema()

    @property
    def expected_columns(self) -> tuple[str, ...]:
        return self.schema.observation_columns

    @property
    def feature_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    def _validated_arrays(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        missing = [column for column in self.expected_columns if column not in frame.columns]
        if missing:
            preview = ", ".join(missing[:8])
            suffix = " ..." if len(missing) > 8 else ""
            raise ValueError(f"Missing Study 1-R observation columns: {preview}{suffix}")

        n_rows = len(frame)
        values = np.empty(
            (
                n_rows,
                len(self.schema.lags),
                len(self.schema.modalities),
                len(self.schema.semantic_variables),
            ),
            dtype=np.float64,
        )
        quality = np.empty(
            (n_rows, len(self.schema.lags), len(self.schema.modalities)),
            dtype=np.float64,
        )
        available = np.empty_like(quality)
        for lag_index, lag in enumerate(self.schema.lags):
            for modality_index, modality in enumerate(self.schema.modalities):
                quality[:, lag_index, modality_index] = pd.to_numeric(
                    frame[f"q_{modality}_{lag}"], errors="coerce"
                ).to_numpy(dtype=np.float64)
                available[:, lag_index, modality_index] = pd.to_numeric(
                    frame[f"available_{modality}_{lag}"], errors="coerce"
                ).to_numpy(dtype=np.float64)
                for variable_index, variable in enumerate(self.schema.semantic_variables):
                    values[:, lag_index, modality_index, variable_index] = pd.to_numeric(
                        frame[f"{modality}_{lag}_{variable}"], errors="coerce"
                    ).to_numpy(dtype=np.float64)

        if not np.all(np.isfinite(quality)):
            raise ValueError("Quality columns contain non-finite values")
        if np.any((quality < -1.0e-12) | (quality > 1.0 + 1.0e-12)):
            raise ValueError("Quality values must be in [0, 1]")
        if not np.all(np.isfinite(available)):
            raise ValueError("Availability columns contain non-finite values")
        if not np.all(np.isclose(available, 0.0) | np.isclose(available, 1.0)):
            raise ValueError("Availability values must be binary")
        return values, np.clip(quality, 0.0, 1.0), np.rint(available)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def to_json_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "schema": self.schema.to_json_dict()}


class BaselineBEncoder(ObservationEncoder):
    """Flatten quality-gated modality observations without structured fusion."""

    kind = "baseline_b"

    @property
    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for lag in self.schema.lags:
            for modality in self.schema.modalities:
                names.extend(
                    f"gated_{modality}_{lag}_{variable}"
                    for variable in self.schema.semantic_variables
                )
                names.append(f"effective_q_{modality}_{lag}")
                names.append(f"available_{modality}_{lag}")
        return tuple(names)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values, quality, available = self._validated_arrays(frame)
        finite = np.isfinite(values)
        effective_q = quality * available
        gated = np.where(finite, values, 0.0) * effective_q[..., None]
        blocks: list[np.ndarray] = []
        for lag_index in range(len(self.schema.lags)):
            for modality_index in range(len(self.schema.modalities)):
                blocks.append(gated[:, lag_index, modality_index, :])
                blocks.append(effective_q[:, lag_index, modality_index, None])
                blocks.append(available[:, lag_index, modality_index, None])
        encoded = np.concatenate(blocks, axis=1)
        if encoded.shape[1] != len(self.feature_names):
            raise RuntimeError("Baseline feature-name and matrix dimensions disagree")
        return encoded


class CompactStructuredEncoder(ObservationEncoder):
    """The compact 49-feature encoder used in frozen Study 1-R.

    Study 1-R2 retains this path, but no longer forces it to replace the raw
    quality-gated observations.  Keeping the original encoder as an explicit
    sub-path makes the architectural change auditable.
    """

    kind = "structured_compact_v1"

    @property
    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for lag in self.schema.lags:
            names.extend(f"fused_{lag}_{v}" for v in self.schema.semantic_variables)
            names.extend(f"disagreement_{lag}_{v}" for v in self.schema.semantic_variables)
            names.extend(f"support_{lag}_{v}" for v in self.schema.semantic_variables)
        names.extend(f"temporal_delta_{v}" for v in self.schema.semantic_variables)
        for lag in self.schema.lags:
            names.extend(
                (
                    f"q_mean_{lag}",
                    f"q_min_{lag}",
                    f"q_max_{lag}",
                    f"q_std_{lag}",
                    f"available_fraction_{lag}",
                )
            )
        names.extend(
            (
                "physics_ttc_s",
                "physics_closing_gap_ratio",
                "physics_crossing_closing_interaction",
                "physics_urgency_crossing_interaction",
            )
        )
        return tuple(names)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values, quality, available = self._validated_arrays(frame)
        finite = np.isfinite(values)
        effective_q = quality * available
        weights = effective_q[..., None] * finite.astype(np.float64)
        safe_values = np.where(finite, values, 0.0)
        weight_sum = weights.sum(axis=2)
        fused = np.divide(
            (weights * safe_values).sum(axis=2),
            weight_sum,
            out=np.zeros_like(weight_sum),
            where=weight_sum > EPS,
        )
        centered = safe_values - fused[:, :, None, :]
        variance = np.divide(
            (weights * centered**2).sum(axis=2),
            weight_sum,
            out=np.zeros_like(weight_sum),
            where=weight_sum > EPS,
        )
        disagreement = np.sqrt(np.maximum(variance, 0.0))
        support = weight_sum / max(1, len(self.schema.modalities))

        blocks: list[np.ndarray] = []
        for lag_index in range(len(self.schema.lags)):
            blocks.extend(
                (
                    fused[:, lag_index, :],
                    disagreement[:, lag_index, :],
                    support[:, lag_index, :],
                )
            )
        blocks.append(fused[:, 1, :] - fused[:, 0, :])

        for lag_index in range(len(self.schema.lags)):
            q = effective_q[:, lag_index, :]
            blocks.append(
                np.column_stack(
                    (
                        q.mean(axis=1),
                        q.min(axis=1),
                        q.max(axis=1),
                        q.std(axis=1),
                        available[:, lag_index, :].mean(axis=1),
                    )
                )
            )

        variable_index = {
            variable: index for index, variable in enumerate(self.schema.semantic_variables)
        }

        def current(name: str, default: float = 0.0) -> np.ndarray:
            index = variable_index.get(name)
            if index is None:
                return np.full(len(frame), default, dtype=np.float64)
            return fused[:, 1, index]

        gap = np.maximum(current("gap_m"), 0.0)
        closing = current("closing_speed_mps")
        crossing = np.clip(current("crossing_risk"), 0.0, 1.0)
        urgency = np.clip(current("route_urgency"), 0.0, 1.0)
        positive_closing = np.maximum(closing, 0.0)
        ttc = np.clip(gap / np.maximum(positive_closing, 0.1), 0.0, 20.0)
        closing_gap = positive_closing / np.maximum(gap, 0.5)
        physics = np.column_stack(
            (
                ttc,
                closing_gap,
                crossing * positive_closing,
                urgency * crossing,
            )
        )
        blocks.append(physics)
        encoded = np.concatenate(blocks, axis=1)
        if encoded.shape[1] != len(self.feature_names):
            raise RuntimeError("Structured feature-name and matrix dimensions disagree")
        if not np.all(np.isfinite(encoded)):
            raise ValueError("Structured encoding produced non-finite values")
        return encoded


class StructuredEncoder(ObservationEncoder):
    """Exploratory dual-path physics-informed encoder for Study 1-R2.

    The encoder receives exactly the same method-blind observation columns as
    Baseline B.  It concatenates (i) the complete Baseline-B quality-gated raw
    path, (ii) the frozen Study-1R compact structured path, and (iii) robust
    inverse-noise fusion and observation-only physics margins.  The constants
    below are frozen properties of the controlled-synthetic observation and
    oracle specifications; no final-test output is used to construct them.

    This is intentionally a higher-capacity candidate.  Study 1-R2 therefore
    evaluates a new dual-path model design, not a parameter-matched repeat of
    the original representation-only comparison.
    """

    kind = "structured_r2_dual_path_v1"

    _BASE_NOISE_SD = np.asarray(
        [
            [2.00, 1.10, 0.60, 0.10, 0.10],  # camera
            [0.70, 0.55, 0.45, 0.18, 0.15],  # lidar
            [1.30, 0.25, 0.55, 0.22, 0.20],  # radar
            [4.00, 1.80, 0.08, 0.16, 0.04],  # ego
        ],
        dtype=np.float64,
    )
    _TTC_CAP_S = 20.0
    _COMFORTABLE_DECELERATION_MPS2 = 4.5
    _STANDSTILL_GAP_M = 2.0
    _REACTION_TIME_S = 0.7
    _STOP_TTC_S = 1.20
    _YIELD_TTC_S = 2.40
    _SLOW_TTC_S = 5.00
    _STOP_CROSSING_RISK = 0.88
    _YIELD_CROSSING_RISK = 0.66
    _SLOW_CROSSING_RISK = 0.36
    _YIELD_STOPPING_MARGIN_M = 0.0
    _SLOW_STOPPING_MARGIN_M = 5.0
    _COLLISION_GAP_M = 0.75

    @property
    def feature_names(self) -> tuple[str, ...]:
        baseline = tuple(
            f"raw_path__{name}" for name in BaselineBEncoder(self.schema).feature_names
        )
        compact = tuple(
            f"compact_path__{name}"
            for name in CompactStructuredEncoder(self.schema).feature_names
        )
        robust: list[str] = []
        for lag in self.schema.lags:
            robust.extend(f"robust_fused_{lag}_{v}" for v in self.schema.semantic_variables)
            robust.extend(f"robust_spread_{lag}_{v}" for v in self.schema.semantic_variables)
        robust.extend(f"robust_temporal_delta_{v}" for v in self.schema.semantic_variables)
        robust.extend(
            (
                "proxy_ttc_s",
                "proxy_stopping_margin_m",
                "margin_ttc_stop_s",
                "margin_ttc_yield_s",
                "margin_ttc_slow_s",
                "margin_crossing_stop",
                "margin_crossing_yield",
                "margin_crossing_slow",
                "margin_stopping_yield_m",
                "margin_stopping_slow_m",
                "margin_collision_gap_m",
                "proxy_action_KEEP",
                "proxy_action_SLOW",
                "proxy_action_YIELD",
                "proxy_action_STOP",
            )
        )
        return baseline + compact + tuple(robust)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values, quality, available = self._validated_arrays(frame)
        finite = np.isfinite(values)
        safe_values = np.where(finite, values, 0.0)

        baseline = BaselineBEncoder(self.schema).transform(frame)
        compact = CompactStructuredEncoder(self.schema).transform(frame)

        # The generator scales each modality/variable noise by
        # base_sd / (0.25 + 0.75 * quality).  Its reciprocal variance is the
        # natural reliability weight.  Availability and finite-value masks are
        # applied before normalization.
        quality_factor = 0.25 + 0.75 * np.clip(quality, 0.0, 1.0)
        precision = (
            quality_factor[..., None] ** 2
            / np.maximum(self._BASE_NOISE_SD[None, None, :, :] ** 2, EPS)
        )
        weights = precision * available[..., None] * finite.astype(np.float64)
        weight_sum = weights.sum(axis=2)
        fused = np.divide(
            (weights * safe_values).sum(axis=2),
            weight_sum,
            out=np.zeros_like(weight_sum),
            where=weight_sum > EPS,
        )
        centered = safe_values - fused[:, :, None, :]
        variance = np.divide(
            (weights * centered**2).sum(axis=2),
            weight_sum,
            out=np.zeros_like(weight_sum),
            where=weight_sum > EPS,
        )
        spread = np.sqrt(np.maximum(variance, 0.0))

        robust_blocks: list[np.ndarray] = []
        for lag_index in range(len(self.schema.lags)):
            robust_blocks.extend((fused[:, lag_index, :], spread[:, lag_index, :]))
        robust_blocks.append(fused[:, 1, :] - fused[:, 0, :])

        variable_index = {
            variable: index for index, variable in enumerate(self.schema.semantic_variables)
        }

        def current(name: str, default: float = 0.0) -> np.ndarray:
            index = variable_index.get(name)
            if index is None:
                return np.full(len(frame), default, dtype=np.float64)
            return fused[:, 1, index]

        gap = np.maximum(current("gap_m"), 0.0)
        closing = current("closing_speed_mps")
        speed = np.maximum(current("ego_speed_mps"), 0.0)
        crossing = np.clip(current("crossing_risk"), 0.0, 1.0)
        positive_closing = np.maximum(closing, 0.0)
        ttc = np.full(len(frame), self._TTC_CAP_S, dtype=np.float64)
        closing_mask = closing > 0.1
        ttc[closing_mask] = np.clip(
            gap[closing_mask] / closing[closing_mask], 0.0, self._TTC_CAP_S
        )
        stopping_margin = gap - (
            self._STANDSTILL_GAP_M
            + self._REACTION_TIME_S * speed
            + positive_closing**2 / (2.0 * self._COMFORTABLE_DECELERATION_MPS2)
        )

        proxy = np.zeros(len(frame), dtype=np.int64)
        proxy[
            (ttc < self._SLOW_TTC_S)
            | (stopping_margin < self._SLOW_STOPPING_MARGIN_M)
            | (crossing > self._SLOW_CROSSING_RISK)
        ] = 1
        proxy[
            (ttc < self._YIELD_TTC_S)
            | (stopping_margin < self._YIELD_STOPPING_MARGIN_M)
            | (crossing > self._YIELD_CROSSING_RISK)
        ] = 2
        proxy[
            (ttc < self._STOP_TTC_S)
            | (crossing > self._STOP_CROSSING_RISK)
            | (gap <= self._COLLISION_GAP_M)
        ] = 3
        proxy_one_hot = np.eye(N_CLASSES, dtype=np.float64)[proxy]
        physics = np.column_stack(
            (
                ttc,
                stopping_margin,
                ttc - self._STOP_TTC_S,
                ttc - self._YIELD_TTC_S,
                ttc - self._SLOW_TTC_S,
                self._STOP_CROSSING_RISK - crossing,
                self._YIELD_CROSSING_RISK - crossing,
                self._SLOW_CROSSING_RISK - crossing,
                stopping_margin - self._YIELD_STOPPING_MARGIN_M,
                stopping_margin - self._SLOW_STOPPING_MARGIN_M,
                gap - self._COLLISION_GAP_M,
            )
        )
        robust_blocks.extend((physics, proxy_one_hot))

        encoded = np.concatenate((baseline, compact, *robust_blocks), axis=1)
        if encoded.shape[1] != len(self.feature_names):
            raise RuntimeError("Study 1-R2 feature-name and matrix dimensions disagree")
        if not np.all(np.isfinite(encoded)):
            raise ValueError("Study 1-R2 encoding produced non-finite values")
        return encoded


def build_encoder(
    method: str, schema: FeatureSchema | None = None
) -> ObservationEncoder:
    normalized = method.strip().lower().replace("_", " ")
    if normalized in {"baseline b", "baseline"}:
        return BaselineBEncoder(schema)
    if normalized in {"structured-r2", "structured r2", "structured", "structured fusion"}:
        return StructuredEncoder(schema)
    raise ValueError(f"Unknown Study 1-R method: {method!r}")


def mlp_parameter_count(
    input_dim: int,
    hidden_width: int | Sequence[int],
    classes: int = N_CLASSES,
) -> int:
    """Parameter count for a fully connected multiclass MLP.

    ``hidden_width`` accepts an integer for the one-layer helper API or the
    frozen protocol's complete sequence (currently ``(48, 24)``).
    """

    hidden = (
        (int(hidden_width),)
        if isinstance(hidden_width, (int, np.integer))
        else tuple(int(value) for value in hidden_width)
    )
    if input_dim < 1 or not hidden or any(value < 1 for value in hidden) or classes < 2:
        raise ValueError("Network dimensions must be positive")
    dimensions = (int(input_dim), *hidden, int(classes))
    return int(
        sum(
            dimensions[index] * dimensions[index + 1] + dimensions[index + 1]
            for index in range(len(dimensions) - 1)
        )
    )


def matched_hidden_width(
    *,
    input_dim: int,
    reference_input_dim: int,
    reference_hidden_width: int,
    trailing_hidden_layers: Sequence[int] = (),
    classes: int = N_CLASSES,
    minimum: int = 2,
    maximum: int = 512,
) -> int:
    """Return the integer width whose parameter count is closest to reference."""

    trailing = tuple(int(value) for value in trailing_hidden_layers)
    target = mlp_parameter_count(
        reference_input_dim,
        (reference_hidden_width, *trailing),
        classes,
    )
    candidates = np.arange(minimum, maximum + 1, dtype=int)
    counts = np.asarray(
        [
            mlp_parameter_count(input_dim, (int(width), *trailing), classes)
            for width in candidates
        ]
    )
    distances = np.abs(counts - target)
    # np.argmin deterministically selects the smaller width on an exact tie.
    return int(candidates[int(np.argmin(distances))])


@dataclass
class Study1RClassifier:
    """An encoder, fitted scaler, MLP and validation-fitted temperature."""

    method: str
    encoder: ObservationEncoder
    estimator: MLPClassifier
    scaler: StandardScaler = field(default_factory=StandardScaler)
    temperature: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def encoded(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.encoder.transform(frame)
        if not hasattr(self.scaler, "mean_"):
            raise RuntimeError("StandardScaler has not been fitted")
        scaled = self.scaler.transform(matrix)
        if not np.all(np.isfinite(scaled)):
            raise ValueError("Scaled model input contains non-finite values")
        return scaled

    def _forward_logits(self, scaled: np.ndarray) -> np.ndarray:
        if not hasattr(self.estimator, "coefs_"):
            raise RuntimeError("MLPClassifier has not been fitted")
        activation = np.asarray(scaled, dtype=np.float64)
        for layer_index, (weights, bias) in enumerate(
            zip(self.estimator.coefs_, self.estimator.intercepts_)
        ):
            activation = activation @ weights + bias
            if layer_index == len(self.estimator.coefs_) - 1:
                break
            name = self.estimator.activation
            if name == "relu":
                activation = np.maximum(activation, 0.0)
            elif name == "tanh":
                activation = np.tanh(activation)
            elif name == "logistic":
                activation = 1.0 / (1.0 + np.exp(-np.clip(activation, -40.0, 40.0)))
            elif name != "identity":
                raise ValueError(f"Unsupported hidden activation: {name}")
        if activation.ndim != 2 or activation.shape[1] != N_CLASSES:
            raise RuntimeError(
                f"Expected {N_CLASSES} output logits, found shape {activation.shape}"
            )
        return activation

    def predict_logits(self, frame: pd.DataFrame) -> np.ndarray:
        """Return uncalibrated raw logits with shape ``[N, 4]``."""

        return self._forward_logits(self.encoded(frame))

    def predict_proba(self, frame: pd.DataFrame, *, calibrated: bool = True) -> np.ndarray:
        logits = self.predict_logits(frame)
        temperature = self.temperature if calibrated else 1.0
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("Temperature must be finite and positive")
        return _softmax(logits / temperature)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(frame), axis=1).astype(np.int64)

    def fit_temperature(
        self,
        frame: pd.DataFrame,
        labels: Iterable[int],
        *,
        lower_bound: float = 0.5,
        upper_bound: float = 5.0,
    ) -> float:
        """Fit one positive temperature on the fixed validation set only."""

        y = np.asarray(list(labels), dtype=np.int64)
        logits = self.predict_logits(frame)
        if len(y) != len(logits):
            raise ValueError("Temperature labels and validation rows differ")

        if not (0.0 < lower_bound < upper_bound):
            raise ValueError("Temperature bounds must satisfy 0 < lower < upper")

        def objective(log_temperature: float) -> float:
            return multiclass_nll(y, _softmax(logits / math.exp(log_temperature)))

        result = minimize_scalar(
            objective,
            method="bounded",
            bounds=(math.log(lower_bound), math.log(upper_bound)),
            options={"xatol": 1.0e-10, "maxiter": 500},
        )
        if not result.success or not math.isfinite(float(result.x)):
            raise RuntimeError(f"Temperature fitting failed: {result.message}")
        self.temperature = float(math.exp(float(result.x)))
        return self.temperature

    def parameter_count(self) -> int:
        if not hasattr(self.estimator, "coefs_"):
            hidden = tuple(int(value) for value in self.estimator.hidden_layer_sizes)
            return mlp_parameter_count(len(self.encoder.feature_names), hidden, N_CLASSES)
        return int(
            sum(array.size for array in self.estimator.coefs_)
            + sum(array.size for array in self.estimator.intercepts_)
        )

    def to_checkpoint_dict(self) -> dict[str, Any]:
        if not hasattr(self.estimator, "coefs_") or not hasattr(self.scaler, "mean_"):
            raise RuntimeError("Cannot checkpoint an unfitted model")
        params = self.estimator.get_params(deep=False)
        allowed_params = {
            "activation",
            "alpha",
            "batch_size",
            "beta_1",
            "beta_2",
            "early_stopping",
            "epsilon",
            "hidden_layer_sizes",
            "learning_rate",
            "learning_rate_init",
            "max_fun",
            "max_iter",
            "momentum",
            "n_iter_no_change",
            "nesterovs_momentum",
            "power_t",
            "random_state",
            "shuffle",
            "solver",
            "tol",
            "validation_fraction",
            "verbose",
            "warm_start",
        }
        serializable_params: dict[str, Any] = {}
        for key in sorted(allowed_params):
            value = params[key]
            if isinstance(value, tuple):
                value = list(value)
            if isinstance(value, np.generic):
                value = value.item()
            serializable_params[key] = value
        n_seen = np.asarray(self.scaler.n_samples_seen_)
        payload = {
            "schema": "egms-drive-study1r-checkpoint-1.0",
            "method": self.method,
            "encoder": self.encoder.to_json_dict(),
            "feature_names": list(self.encoder.feature_names),
            "classes": [int(value) for value in np.asarray(self.estimator.classes_)],
            "scaler": {
                "mean": _finite_float_list(self.scaler.mean_),
                "scale": _finite_float_list(self.scaler.scale_),
                "var": _finite_float_list(self.scaler.var_),
                "n_samples_seen": n_seen.tolist(),
            },
            "estimator_params": serializable_params,
            "coefs": [_finite_float_list(value) for value in self.estimator.coefs_],
            "intercepts": [
                _finite_float_list(value) for value in self.estimator.intercepts_
            ],
            "temperature": float(self.temperature),
            "parameter_count": self.parameter_count(),
            "metadata": self.metadata,
        }
        # Enforce strict JSON compatibility before the payload can be written.
        json.dumps(payload, allow_nan=False, sort_keys=True)
        return payload

    @classmethod
    def from_checkpoint_dict(cls, payload: Mapping[str, Any]) -> "Study1RClassifier":
        if payload.get("schema") != "egms-drive-study1r-checkpoint-1.0":
            raise ValueError("Unsupported Study 1-R checkpoint schema")
        encoder_payload = payload["encoder"]
        schema = FeatureSchema.from_json_dict(encoder_payload["schema"])
        encoder = build_encoder(str(payload["method"]), schema)
        if encoder.kind != encoder_payload["kind"]:
            raise ValueError("Checkpoint method and encoder kind disagree")
        if list(encoder.feature_names) != list(payload["feature_names"]):
            raise ValueError("Checkpoint feature names disagree with reconstructed encoder")

        params = dict(payload["estimator_params"])
        params["hidden_layer_sizes"] = tuple(int(value) for value in params["hidden_layer_sizes"])
        estimator = MLPClassifier(**params)
        classes = np.asarray(payload["classes"], dtype=np.int64)
        if not np.array_equal(classes, np.arange(N_CLASSES, dtype=np.int64)):
            raise ValueError("Checkpoint must contain ordered classes [0, 1, 2, 3]")
        # Initialize sklearn's fitted attributes without relying on pickle, then
        # replace the random arrays by the audited JSON values.
        input_dim = len(encoder.feature_names)
        dummy_x = np.zeros((N_CLASSES, input_dim), dtype=np.float64)
        estimator.partial_fit(dummy_x, classes, classes=classes)
        coefs = [np.asarray(value, dtype=np.float64) for value in payload["coefs"]]
        intercepts = [
            np.asarray(value, dtype=np.float64) for value in payload["intercepts"]
        ]
        if [array.shape for array in coefs] != [array.shape for array in estimator.coefs_]:
            raise ValueError("Checkpoint coefficient shapes do not match estimator topology")
        if [array.shape for array in intercepts] != [
            array.shape for array in estimator.intercepts_
        ]:
            raise ValueError("Checkpoint intercept shapes do not match estimator topology")
        estimator.coefs_ = coefs
        estimator.intercepts_ = intercepts

        scaler_payload = payload["scaler"]
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(scaler_payload["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(scaler_payload["scale"], dtype=np.float64)
        scaler.var_ = np.asarray(scaler_payload["var"], dtype=np.float64)
        n_seen = np.asarray(scaler_payload["n_samples_seen"])
        scaler.n_samples_seen_ = n_seen.item() if n_seen.ndim == 0 else n_seen
        scaler.n_features_in_ = input_dim
        if any(len(value) != input_dim for value in (scaler.mean_, scaler.scale_, scaler.var_)):
            raise ValueError("Checkpoint scaler dimension does not match encoder")

        model = cls(
            method=str(payload["method"]),
            encoder=encoder,
            estimator=estimator,
            scaler=scaler,
            temperature=float(payload["temperature"]),
            metadata=dict(payload.get("metadata", {})),
        )
        if model.parameter_count() != int(payload["parameter_count"]):
            raise ValueError("Checkpoint parameter count is inconsistent")
        return model

    def save_checkpoint(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(
            self.to_checkpoint_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        destination.write_text(text + "\n", encoding="utf-8", newline="\n")
        return destination

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "Study1RClassifier":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint root must be a JSON object")
        return cls.from_checkpoint_dict(payload)


def make_classifier(
    *,
    method: str,
    encoder: ObservationEncoder,
    hidden_layer_sizes: Sequence[int],
    learning_rate_init: float,
    alpha: float,
    random_state: int,
) -> Study1RClassifier:
    """Construct an unfitted frozen-head partial-fit classifier."""

    hidden = tuple(int(value) for value in hidden_layer_sizes)
    if not hidden or any(value < 2 for value in hidden):
        raise ValueError("Every hidden-layer width must be at least 2")
    estimator = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=float(alpha),
        batch_size="auto",  # Batching is controlled explicitly by training.py.
        learning_rate="constant",
        learning_rate_init=float(learning_rate_init),
        max_iter=1,
        shuffle=False,
        random_state=int(random_state),
        tol=0.0,
        n_iter_no_change=10_000,
        early_stopping=False,
        warm_start=False,
    )
    return Study1RClassifier(method=method, encoder=encoder, estimator=estimator)


__all__ = [
    "ACTION_NAMES",
    "N_CLASSES",
    "MODEL_BASELINE_B",
    "MODEL_STRUCTURED",
    "MODEL_SLUGS",
    "FeatureSchema",
    "ObservationEncoder",
    "BaselineBEncoder",
    "CompactStructuredEncoder",
    "StructuredEncoder",
    "Study1RClassifier",
    "build_encoder",
    "make_classifier",
    "matched_hidden_width",
    "mlp_parameter_count",
    "multiclass_nll",
]
