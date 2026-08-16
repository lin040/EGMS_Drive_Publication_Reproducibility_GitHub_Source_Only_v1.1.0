from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize_scalar


EPS = 1.0e-12


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(np.asarray(array))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponents = np.exp(shifted)
    return exponents / np.maximum(exponents.sum(axis=axis, keepdims=True), EPS)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def _probabilities(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("probabilities must be [n, classes]")
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities contain non-finite values")
    if np.any(p < -1.0e-10) or np.any(p > 1.0 + 1.0e-10):
        raise ValueError("probability outside [0,1]")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1.0e-8, rtol=0.0):
        raise ValueError("probability rows do not sum to one")
    return np.clip(p, 0.0, 1.0)


def confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], n_classes: int) -> np.ndarray:
    truth = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matrix, (truth, pred), 1)
    return matrix


def macro_f1(y_true: Iterable[int], y_pred: Iterable[int], n_classes: int) -> float:
    matrix = confusion_matrix(y_true, y_pred, n_classes).astype(float)
    scores = []
    for label in range(n_classes):
        tp = matrix[label, label]
        fp = matrix[:, label].sum() - tp
        fn = matrix[label, :].sum() - tp
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def negative_log_likelihood(y_true: Iterable[int], probabilities: np.ndarray) -> float:
    truth = np.asarray(list(y_true), dtype=int)
    p = _probabilities(probabilities)
    return float(np.mean(-np.log(np.maximum(p[np.arange(len(truth)), truth], EPS))))


def multiclass_brier(y_true: Iterable[int], probabilities: np.ndarray) -> float:
    truth = np.asarray(list(y_true), dtype=int)
    p = _probabilities(probabilities)
    one_hot = np.eye(p.shape[1], dtype=float)[truth]
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def calibration_bins(
    y_true: Iterable[int], probabilities: np.ndarray, n_bins: int = 15
) -> pd.DataFrame:
    truth = np.asarray(list(y_true), dtype=int)
    p = _probabilities(probabilities)
    confidence = p.max(axis=1)
    prediction = p.argmax(axis=1)
    correctness = (prediction == truth).astype(float)
    # Equal-mass bins reduce the large finite-sample bias that sparse fixed
    # width bins can introduce. Stable sorting makes tied-confidence handling
    # deterministic. Each observation belongs to exactly one bin.
    ordered = np.argsort(confidence, kind="stable")
    groups = np.array_split(ordered, int(n_bins))
    rows = []
    for bin_index, indices in enumerate(groups):
        active = len(indices) > 0
        rows.append(
            {
                "bin": bin_index + 1,
                "lower": float(confidence[indices].min()) if active else np.nan,
                "upper": float(confidence[indices].max()) if active else np.nan,
                "count": int(len(indices)),
                "mean_confidence": float(confidence[indices].mean()) if active else np.nan,
                "accuracy": float(correctness[indices].mean()) if active else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: Iterable[int], probabilities: np.ndarray, n_bins: int = 15
) -> float:
    bins = calibration_bins(y_true, probabilities, n_bins)
    total = bins["count"].sum()
    if total == 0:
        return float("nan")
    gap = (bins["accuracy"] - bins["mean_confidence"]).abs().fillna(0.0)
    return float(np.sum((bins["count"] / total) * gap))


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, probabilities: np.ndarray, labels: Iterable[int]) -> "TemperatureScaler":
        p = _probabilities(probabilities)
        y = np.asarray(list(labels), dtype=int)
        logits = np.log(np.maximum(p, EPS))

        def objective(log_temperature: float) -> float:
            calibrated = softmax(logits / np.exp(log_temperature), axis=1)
            return negative_log_likelihood(y, calibrated)

        result = minimize_scalar(objective, bounds=(-2.5, 2.5), method="bounded")
        self.temperature = float(np.exp(result.x)) if result.success else 1.0
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        p = _probabilities(probabilities)
        return softmax(np.log(np.maximum(p, EPS)) / self.temperature, axis=1)


def seed_summary(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    estimate = float(x.mean())
    if len(x) == 1:
        return (estimate, float("nan"), float("nan"))
    from scipy.stats import t

    half = float(t.ppf((1 + confidence) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return (estimate, estimate - half, estimate + half)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    count = len(p)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def software_manifest() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def write_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
