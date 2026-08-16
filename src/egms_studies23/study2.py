from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from time import perf_counter_ns
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .common import (
    TemperatureScaler,
    array_digest,
    calibration_bins,
    expected_calibration_error,
    macro_f1,
    multiclass_brier,
    negative_log_likelihood,
    normalize_rows,
    softmax,
    stable_seed,
)


ACTION_NAMES = ("KEEP", "SLOW", "YIELD", "STOP")
SCENARIOS = ("cut_in", "platoon_brake", "occluded_crossing", "right_turn", "merge")


def _orthogonal_projection(rng: np.random.Generator, latent_dim: int, raw_dim: int) -> np.ndarray:
    matrix = rng.normal(size=(raw_dim, raw_dim))
    q, _ = np.linalg.qr(matrix)
    return q[:latent_dim, :]


def _weather_for(split: str, index: int) -> str:
    if split == "test":
        return "clear"
    if split == "stress":
        return ("night", "rain", "fog", "snow")[index % 4]
    return ("clear", "night", "rain", "fog")[index % 4]


def _quality_profile(weather: str) -> dict[str, float]:
    return {
        "clear": {"camera": 0.92, "lidar": 0.92, "radar": 0.88, "ego": 0.98},
        "night": {"camera": 0.55, "lidar": 0.90, "radar": 0.87, "ego": 0.98},
        "rain": {"camera": 0.48, "lidar": 0.68, "radar": 0.82, "ego": 0.97},
        "fog": {"camera": 0.34, "lidar": 0.55, "radar": 0.84, "ego": 0.97},
        "snow": {"camera": 0.38, "lidar": 0.48, "radar": 0.73, "ego": 0.95},
    }[weather].copy()


def _latent_sequence(
    data_seed: int,
    split: str,
    scene_index: int,
    length: int,
    latent_dim: int,
    training_replicate: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    # Test and stress use the same latent namespace, producing a paired clean
    # and corrupted counterfactual for every shared scene index.
    namespace = "evaluation" if split in {"test", "stress"} else split
    replicate_namespace = training_replicate if split == "train" else "fixed"
    rng = np.random.default_rng(
        stable_seed(data_seed, "latent", namespace, replicate_namespace, scene_index)
    )
    scenario = SCENARIOS[scene_index % len(SCENARIOS)]
    density = ("low", "medium", "high")[(scene_index // len(SCENARIOS)) % 3]
    class_hint = scene_index % 4
    base_risk = (-1.05, -0.20, 0.48, 1.12)[class_hint]
    state = rng.normal(0.0, 0.35, size=latent_dim)
    state[0] = base_risk + rng.normal(0.0, 0.10)
    state[1] = {
        "cut_in": 0.15,
        "platoon_brake": 0.25,
        "occluded_crossing": 0.72,
        "right_turn": 0.88,
        "merge": 0.50,
    }[scenario] + rng.normal(0.0, 0.08)
    state[4] = {"low": -0.65, "medium": 0.0, "high": 0.65}[density]
    latent = np.zeros((length, latent_dim), dtype=float)
    labels = np.zeros(length, dtype=int)
    event_direction = rng.normal(0.0, 0.08, size=latent_dim)
    event_direction[0] += {"cut_in": 0.10, "platoon_brake": 0.16, "occluded_crossing": 0.08, "right_turn": 0.12, "merge": 0.09}[scenario]
    for t in range(length):
        innovation = rng.normal(0.0, 0.045, size=latent_dim)
        state = 0.94 * state + 0.06 * event_direction * t + innovation
        state[2] = np.sin((t + scene_index % 7) / 5.0) + rng.normal(0, 0.04)
        state[3] = 0.50 - 0.22 * state[0] + rng.normal(0, 0.05)
        latent[t] = state
        score = state[0] + 0.33 * state[1] + 0.10 * state[4] - 0.08 * state[3]
        labels[t] = int(np.digitize(score, [-0.48, 0.20, 0.83]))
    return latent, labels, scenario, density


def generate_study2_split(
    cfg: dict, split: str, training_replicate: int | None = None
) -> pd.DataFrame:
    study = cfg["study2"]
    modalities = tuple(cfg["common"]["modalities"])
    count_key = {
        "train": "train_sequences",
        "validation": "validation_sequences",
        "test": "test_sequences",
        "stress": "stress_sequences",
    }[split]
    n_sequences = int(study[count_key])
    length = int(study["sequence_length"])
    latent_dim = int(study["latent_dim"])
    raw_dim = int(study["raw_dim"])
    data_seed = int(study["data_seed"])
    transform_rng = np.random.default_rng(stable_seed(data_seed, "modality_transforms"))
    transforms = {m: _orthogonal_projection(transform_rng, latent_dim, raw_dim) for m in modalities}
    biases = {m: transform_rng.normal(0.0, 0.08, size=raw_dim) for m in modalities}
    rows: list[dict[str, object]] = []
    for scene_index in range(n_sequences):
        latent, labels, scenario, density = _latent_sequence(
            data_seed, split, scene_index, length, latent_dim, training_replicate
        )
        weather = _weather_for(split, scene_index)
        profile = _quality_profile(weather)
        density_penalty = {"low": 0.0, "medium": 0.035, "high": 0.075}[density]
        latent_pair_id = f"eval_{scene_index:05d}" if split in {"test", "stress"} else f"{split}_{scene_index:05d}"
        corruption_rng = np.random.default_rng(
            stable_seed(
                data_seed,
                "corruption",
                split,
                training_replicate if split == "train" else "fixed",
                scene_index,
                weather,
            )
        )
        modality_lag = {
            m: int(corruption_rng.choice([0, 0, 0, 1])) if m != "ego" else 0
            for m in modalities
        }
        burst_start = int(corruption_rng.integers(2, max(3, length - 2)))
        burst_modality = modalities[int(corruption_rng.integers(0, len(modalities) - 1))]
        previous_observation: dict[str, np.ndarray] = {}
        for t in range(length):
            row: dict[str, object] = {
                "split": split,
                "scene_id": f"{split}_{scene_index:05d}",
                "latent_pair_id": latent_pair_id,
                "frame": t,
                "weather": weather,
                "scenario": scenario,
                "density": density,
                "y_true": int(labels[t]),
            }
            for j in range(latent_dim):
                row[f"latent_{j}"] = float(latent[t, j])
            for modality in modalities:
                q = float(np.clip(profile[modality] - density_penalty + corruption_rng.normal(0, 0.035), 0.08, 0.995))
                source_t = max(0, t - modality_lag[modality])
                semantic = latent[source_t]
                noise_sd = 0.035 + 0.28 * (1.0 - q)
                observed = np.tanh(semantic @ transforms[modality] + biases[modality])
                observed += corruption_rng.normal(0.0, noise_sd, size=raw_dim)
                if split == "stress" and modality != "ego":
                    # Controlled injected sensor-domain perturbation. This is
                    # not claimed to be CARLA's natural weather physics.
                    observed += 0.055 * np.sin(np.arange(raw_dim) + scene_index % 5)
                available = 1
                dropout_probability = 0.01 + 0.10 * (1.0 - q)
                if corruption_rng.random() < dropout_probability:
                    available = 0
                if split == "stress" and modality == burst_modality and burst_start <= t < burst_start + 2:
                    available = 0
                if available == 0:
                    observed[:] = 0.0
                # Health descriptors use sensor observations only. Hidden DGM
                # corruption scale, lag, and q never enter these features.
                signal_energy = float(np.mean(np.abs(observed)))
                observed_spread = float(np.std(observed))
                roughness = float(np.mean(np.abs(np.diff(observed)))) if raw_dim > 1 else 0.0
                if modality in previous_observation:
                    temporal_change = float(
                        np.linalg.norm(observed - previous_observation[modality])
                        / np.sqrt(raw_dim)
                    )
                    temporal_stability = float(np.exp(-temporal_change))
                else:
                    temporal_stability = 1.0
                previous_observation[modality] = observed.copy()
                health = np.array(
                    [
                        signal_energy,
                        float(available),
                        observed_spread + 0.25 * roughness,
                        temporal_stability,
                    ]
                )
                row[f"quality_true_audit_{modality}"] = q
                row[f"available_{modality}"] = available
                for j, value in enumerate(health):
                    row[f"health_{modality}_{j}"] = float(value)
                for j, value in enumerate(observed):
                    row[f"{modality}_{j}"] = float(value)
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["scene_id", "frame"], kind="stable").reset_index(drop=True)
    if set(frame["y_true"].unique()) != set(range(4)):
        raise RuntimeError(f"Study 2 {split} split lacks one or more action classes")
    return frame


class MultiviewAligner:
    def __init__(self, modalities: tuple[str, ...], raw_dim: int, embedding_dim: int, enabled: bool, seed: int):
        self.modalities = modalities
        self.raw_dim = raw_dim
        self.embedding_dim = embedding_dim
        self.enabled = enabled
        self.seed = seed
        self.scalers: dict[str, StandardScaler] = {}
        self.regressors: dict[str, Ridge] = {}
        self.pca: PCA | None = None

    def columns(self, modality: str) -> list[str]:
        return [f"{modality}_{j}" for j in range(self.raw_dim)]

    def fit(self, frame: pd.DataFrame) -> "MultiviewAligner":
        standardized = []
        for modality in self.modalities:
            scaler = StandardScaler().fit(frame[self.columns(modality)].to_numpy(float))
            self.scalers[modality] = scaler
            standardized.append(scaler.transform(frame[self.columns(modality)].to_numpy(float)))
        if self.enabled:
            concatenated = np.concatenate(standardized, axis=1)
            self.pca = PCA(n_components=self.embedding_dim, whiten=True, random_state=self.seed)
            anchor = self.pca.fit_transform(concatenated)
            for modality, values in zip(self.modalities, standardized):
                self.regressors[modality] = Ridge(alpha=1.0).fit(values, anchor)
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        outputs = {}
        for modality in self.modalities:
            values = self.scalers[modality].transform(frame[self.columns(modality)].to_numpy(float))
            if self.enabled:
                embedded = self.regressors[modality].predict(values)
            else:
                embedded = values[:, : self.embedding_dim]
                if embedded.shape[1] < self.embedding_dim:
                    embedded = np.pad(embedded, ((0, 0), (0, self.embedding_dim - embedded.shape[1])))
            outputs[modality] = embedded
        return outputs


class ReliabilityEstimator:
    def __init__(self, modalities: tuple[str, ...], temperature: float):
        self.modalities = modalities
        self.temperature = float(temperature)
        self.models: dict[str, Ridge] = {}

    @staticmethod
    def health_columns(modality: str) -> list[str]:
        return [f"health_{modality}_{j}" for j in range(4)]

    def fit(self, frame: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> "ReliabilityEstimator":
        stack = np.stack([embeddings[m] for m in self.modalities], axis=1)
        anchor = normalize_rows(stack.mean(axis=1))
        for modality in self.modalities:
            target = np.sum(normalize_rows(embeddings[modality]) * anchor, axis=1)
            self.models[modality] = Ridge(alpha=3.0).fit(
                frame[self.health_columns(modality)].to_numpy(float), target
            )
        return self

    def scores(self, frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                self.models[m].predict(frame[self.health_columns(m)].to_numpy(float))
                for m in self.modalities
            ]
        ) * self.temperature


class TemporalCompensator:
    def __init__(self, strength: float, embedding_dim: int):
        self.strength = float(strength)
        self.embedding_dim = embedding_dim
        self.delta_model = Ridge(alpha=2.0)

    def fit(self, frame: pd.DataFrame, fused: np.ndarray, ego_embedding: np.ndarray) -> "TemporalCompensator":
        inputs = []
        targets = []
        for _, indices in frame.groupby("scene_id", sort=False).indices.items():
            idx = np.asarray(indices)
            if len(idx) < 2:
                continue
            inputs.append(ego_embedding[idx[1:]] - ego_embedding[idx[:-1]])
            targets.append(fused[idx[1:]] - fused[idx[:-1]])
        self.delta_model.fit(np.concatenate(inputs), np.concatenate(targets))
        return self

    def apply(self, frame: pd.DataFrame, fused: np.ndarray, ego_embedding: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
        output = fused.copy()
        prediction_errors = np.full(len(frame), np.nan, dtype=float)
        for _, indices in frame.groupby("scene_id", sort=False).indices.items():
            idx = np.asarray(indices)
            for position in range(1, len(idx)):
                previous = idx[position - 1]
                current = idx[position]
                delta = self.delta_model.predict((ego_embedding[current] - ego_embedding[previous])[None, :])[0]
                warped_previous = output[previous] + delta
                # Held-out one-step prediction error is measured against the
                # current unsmoothed observation. Unlike ||output-warp||, this
                # quantity is not algebraically forced to favor smoothing.
                prediction_errors[current] = float(
                    np.linalg.norm(warped_previous - fused[current])
                )
                if enabled:
                    output[current] = (1.0 - self.strength) * fused[current] + self.strength * warped_previous
        return output, prediction_errors


@dataclass(frozen=True)
class Study2Condition:
    name: str
    alignment: bool = True
    reliability: bool = True
    temporal: bool = True
    dropout_distillation: bool = True


CONDITIONS = {
    "Full": Study2Condition("Full"),
    "Equal weighting": Study2Condition("Equal weighting", reliability=False),
    "No alignment": Study2Condition("No alignment", alignment=False),
    "No temporal consistency": Study2Condition("No temporal consistency", temporal=False),
    "No modality dropout-distillation": Study2Condition(
        "No modality dropout-distillation", dropout_distillation=False
    ),
}


class Study2Model:
    def __init__(self, cfg: dict, condition: Study2Condition, training_seed: int):
        study = cfg["study2"]
        self.cfg = cfg
        self.condition = condition
        self.training_seed = int(training_seed)
        self.modalities = tuple(cfg["common"]["modalities"])
        self.raw_dim = int(study["raw_dim"])
        self.embedding_dim = int(study["embedding_dim"])
        self.aligner = MultiviewAligner(
            self.modalities, self.raw_dim, self.embedding_dim, condition.alignment, training_seed
        )
        self.reliability = ReliabilityEstimator(
            self.modalities, float(study["reliability_temperature"])
        )
        self.temporal = TemporalCompensator(float(study["temporal_strength"]), self.embedding_dim)
        self.distiller: Ridge | None = None
        self.classifier = LogisticRegression(
            C=float(study["classifier_C"]),
            max_iter=int(study["classifier_max_iter"]),
            class_weight="balanced",
            random_state=training_seed,
            solver="lbfgs",
        )
        self.temperature = TemperatureScaler()

    def _available(self, frame: pd.DataFrame, missing_pattern: str = "none") -> np.ndarray:
        available = frame[[f"available_{m}" for m in self.modalities]].to_numpy(float)
        if missing_pattern != "none":
            for modality in missing_pattern.split("+"):
                available[:, self.modalities.index(modality)] = 0.0
        return available

    def _base_fusion(
        self, frame: pd.DataFrame, embeddings: dict[str, np.ndarray], missing_pattern: str
    ) -> tuple[np.ndarray, np.ndarray]:
        stack = np.stack([embeddings[m] for m in self.modalities], axis=1)
        available = self._available(frame, missing_pattern)
        if self.condition.reliability:
            scores = self.reliability.scores(frame)
        else:
            scores = np.zeros_like(available)
        scores = np.where(available > 0.5, scores, -1.0e9)
        all_missing = available.sum(axis=1) == 0
        if all_missing.any():
            scores[all_missing, self.modalities.index("ego")] = 0.0
        weights = softmax(scores, axis=1)
        fused = np.sum(weights[:, :, None] * stack, axis=1)
        return fused, weights

    def _represent(
        self, frame: pd.DataFrame, missing_pattern: str = "none", apply_distiller: bool = True
    ) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
        embeddings = self.aligner.transform(frame)
        base, weights = self._base_fusion(frame, embeddings, missing_pattern)
        fused, drift = self.temporal.apply(
            frame, base, embeddings["ego"], enabled=self.condition.temporal
        )
        if (
            apply_distiller
            and self.condition.dropout_distillation
            and self.distiller is not None
            and missing_pattern != "none"
        ):
            fused = self.distiller.predict(fused)
        return fused, embeddings, weights, drift

    def fit(self, training: pd.DataFrame, validation: pd.DataFrame) -> "Study2Model":
        self.aligner.fit(training)
        train_embeddings = self.aligner.transform(training)
        self.reliability.fit(training, train_embeddings)
        train_base, _ = self._base_fusion(training, train_embeddings, "none")
        self.temporal.fit(training, train_base, train_embeddings["ego"])
        train_full, _, _, _ = self._represent(training, "none", apply_distiller=False)
        x_train = train_full
        y_train = training["y_true"].to_numpy(int)
        if self.condition.dropout_distillation:
            rng = np.random.default_rng(stable_seed(self.training_seed, "modality_dropout"))
            patterns = ("camera", "lidar", "radar", "ego", "camera+lidar")
            masked_blocks = []
            teacher_blocks = []
            label_blocks = []
            scene_ids = training["scene_id"].drop_duplicates().to_numpy()
            selected_scenes = set(
                rng.choice(
                    scene_ids,
                    size=max(1, int(len(scene_ids) * float(self.cfg["study2"]["modality_dropout_probability"]))),
                    replace=False,
                )
            )
            for pattern_index, pattern in enumerate(patterns):
                selected = training["scene_id"].isin(
                    [scene for i, scene in enumerate(sorted(selected_scenes)) if i % len(patterns) == pattern_index]
                )
                if not selected.any():
                    continue
                block = training.loc[selected].copy().reset_index(drop=True)
                masked, _, _, _ = self._represent(block, pattern, apply_distiller=False)
                teacher, _, _, _ = self._represent(block, "none", apply_distiller=False)
                masked_blocks.append(masked)
                teacher_blocks.append(teacher)
                label_blocks.append(block["y_true"].to_numpy(int))
            if masked_blocks:
                masked_all = np.concatenate(masked_blocks)
                teacher_all = np.concatenate(teacher_blocks)
                self.distiller = Ridge(alpha=4.0).fit(masked_all, teacher_all)
                distilled = self.distiller.predict(masked_all)
                x_train = np.concatenate([x_train, distilled])
                y_train = np.concatenate([y_train, np.concatenate(label_blocks)])
        self.classifier.fit(x_train, y_train)
        validation_features, _, _, _ = self._represent(validation, "none")
        validation_raw = self.classifier.predict_proba(validation_features)
        self.temperature.fit(validation_raw, validation["y_true"].to_numpy(int))
        return self

    def predict(
        self, frame: pd.DataFrame, missing_pattern: str = "none"
    ) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        features, embeddings, weights, drift = self._represent(frame, missing_pattern)
        probabilities = self.temperature.transform(self.classifier.predict_proba(features))
        return probabilities, embeddings, weights, drift, features


def alignment_metrics(
    frame: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    max_sequences: int,
) -> dict[str, float]:
    final_rows = frame.groupby("scene_id", sort=False).tail(1).head(int(max_sequences))
    indices = final_rows.index.to_numpy()
    camera = normalize_rows(embeddings["camera"][indices])
    lidar = normalize_rows(embeddings["lidar"][indices])
    similarities = camera @ lidar.T
    positive = np.diag(similarities)
    negative = similarities[~np.eye(len(similarities), dtype=bool)]
    negative_mean = float(negative.mean())
    negative_sd = float(negative.std(ddof=1))
    sas = float(np.mean((positive - negative_mean) / max(negative_sd, 1.0e-8)))
    order = np.argsort(-similarities, axis=1)
    ranks = np.argmax(order == np.arange(len(order))[:, None], axis=1) + 1
    reverse_order = np.argsort(-similarities.T, axis=1)
    reverse_ranks = np.argmax(reverse_order == np.arange(len(order))[:, None], axis=1) + 1
    result = {
        "sas": sas,
        "positive_similarity": float(positive.mean()),
        "negative_similarity": negative_mean,
        "embedding_effective_rank_camera": float(np.exp(-np.sum(_spectrum_entropy(camera) * np.log(np.maximum(_spectrum_entropy(camera), 1e-12))))),
    }
    for k in (1, 5, 10):
        result[f"recall_at_{k}"] = float(0.5 * (np.mean(ranks <= k) + np.mean(reverse_ranks <= k)))
    return result


def _spectrum_entropy(values: np.ndarray) -> np.ndarray:
    singular = np.linalg.svd(values - values.mean(axis=0), compute_uv=False)
    proportions = singular / max(singular.sum(), 1.0e-12)
    return proportions


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, bins: int) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    return {
        "macro_f1": macro_f1(labels, prediction, probabilities.shape[1]),
        "nll": negative_log_likelihood(labels, probabilities),
        "brier": multiclass_brier(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities, bins),
    }


def _frame_digest(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    columns = np.frombuffer("\x1f".join(frame.columns).encode("utf-8"), dtype=np.uint8)
    return array_digest(hashed, columns)


def _study2_checkpoint_arrays(model: Study2Model) -> list[np.ndarray]:
    arrays: list[np.ndarray] = [
        model.classifier.coef_,
        model.classifier.intercept_,
        model.classifier.classes_,
        np.array([model.temperature.temperature]),
        model.temporal.delta_model.coef_,
        np.atleast_1d(model.temporal.delta_model.intercept_),
        np.frombuffer(json.dumps(asdict(model.condition), sort_keys=True).encode("utf-8"), dtype=np.uint8),
    ]
    if model.aligner.pca is not None:
        arrays.extend(
            [
                model.aligner.pca.components_,
                model.aligner.pca.mean_,
                model.aligner.pca.explained_variance_,
            ]
        )
    for modality in model.modalities:
        scaler = model.aligner.scalers[modality]
        arrays.extend([scaler.mean_, scaler.scale_])
        if modality in model.aligner.regressors:
            regressor = model.aligner.regressors[modality]
            arrays.extend([regressor.coef_, np.atleast_1d(regressor.intercept_)])
        reliability = model.reliability.models[modality]
        arrays.extend([reliability.coef_, np.atleast_1d(reliability.intercept_)])
    if model.distiller is not None:
        arrays.extend([model.distiller.coef_, np.atleast_1d(model.distiller.intercept_)])
    return arrays


def _prediction_frame(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    training_seed: int,
    method: str,
    domain: str,
    missing_pattern: str,
) -> pd.DataFrame:
    output = source[["scene_id", "frame", "y_true"]].copy()
    output.insert(0, "training_seed", int(training_seed))
    output.insert(1, "method", method)
    output.insert(2, "domain", domain)
    output.insert(3, "missing_pattern", missing_pattern)
    output["y_pred"] = probabilities.argmax(axis=1)
    for index, action in enumerate(ACTION_NAMES):
        output[f"p_{action}"] = probabilities[:, index]
    return output


def _scene_metric_rows(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    drift: np.ndarray,
    training_seed: int,
    method: str,
    domain: str,
    bins: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scene_id, indices in source.groupby("scene_id", sort=False).indices.items():
        idx = np.asarray(indices)
        valid = drift[idx][np.isfinite(drift[idx])]
        rows.append(
            {
                "training_seed": training_seed,
                "method": method,
                "domain": domain,
                "scene_id": scene_id,
                **classification_metrics(source.iloc[idx]["y_true"].to_numpy(int), probabilities[idx], bins),
                "feature_drift": float(valid.mean()) if len(valid) else np.nan,
            }
        )
    return rows


def run_study2(cfg: dict) -> dict[str, pd.DataFrame]:
    fixed_splits = {
        split: generate_study2_split(cfg, split)
        for split in ("validation", "test", "stress")
    }
    methods = list(cfg["study2"]["methods"])
    seeds = list(cfg["common"]["training_seeds"])
    bins = int(cfg["common"]["calibration_bins"])
    result_rows = []
    missing_rows = []
    prediction_frames: list[pd.DataFrame] = []
    scene_rows: list[dict[str, object]] = []
    missing_scene_rows: list[dict[str, object]] = []
    calibration_rows = []
    latency_rows = []
    model_rows = []
    split_rows: list[dict[str, object]] = []
    for name, data in fixed_splits.items():
        split_rows.append(
            {
                "split": name,
                "training_seed": np.nan,
                "scenes": data["scene_id"].nunique(),
                "frames": len(data),
                "latent_pair_ids": data["latent_pair_id"].nunique(),
                "weather_levels": ",".join(sorted(data["weather"].unique())),
                "action_classes": ",".join(map(str, sorted(data["y_true"].unique()))),
                "data_sha256": _frame_digest(data),
            }
        )
    for training_seed in seeds:
        training = generate_study2_split(cfg, "train", int(training_seed))
        training_hash = _frame_digest(training)
        split_rows.append(
            {
                "split": "train",
                "training_seed": training_seed,
                "scenes": training["scene_id"].nunique(),
                "frames": len(training),
                "latent_pair_ids": training["latent_pair_id"].nunique(),
                "weather_levels": ",".join(sorted(training["weather"].unique())),
                "action_classes": ",".join(map(str, sorted(training["y_true"].unique()))),
                "data_sha256": training_hash,
            }
        )
        for method in methods:
            model = Study2Model(cfg, CONDITIONS[method], int(training_seed)).fit(
                training, fixed_splits["validation"]
            )
            digest_arrays = _study2_checkpoint_arrays(model)
            model_rows.append(
                {
                    "training_seed": training_seed,
                    "method": method,
                    "model_sha256": array_digest(*digest_arrays),
                    "training_data_sha256": training_hash,
                    "numeric_state_values": int(sum(array.size for array in digest_arrays)),
                    "hash_scope": "complete numeric fitted state + condition",
                    "temperature": float(model.temperature.temperature),
                    "alignment_enabled": model.condition.alignment,
                    "reliability_enabled": model.condition.reliability,
                    "temporal_enabled": model.condition.temporal,
                    "dropout_distillation_enabled": model.condition.dropout_distillation,
                }
            )
            for domain in ("test", "stress"):
                frame = fixed_splits[domain]
                domain_name = "clean_controlled" if domain == "test" else "adverse_controlled_injected"
                start = perf_counter_ns()
                probabilities, embeddings, weights, drift, features = model.predict(frame)
                elapsed_ms = (perf_counter_ns() - start) / 1.0e6
                metrics = classification_metrics(frame["y_true"].to_numpy(int), probabilities, bins)
                align = alignment_metrics(
                    frame.reset_index(drop=True),
                    embeddings,
                    int(cfg["study2"]["retrieval_pool_sequences"]),
                )
                valid_drift = drift[np.isfinite(drift)]
                metrics.update(align)
                metrics["feature_drift"] = float(valid_drift.mean())
                metrics["representation_variance"] = float(np.mean(np.var(features, axis=0)))
                metrics["latency_ms_per_frame"] = float(elapsed_ms / len(frame))
                result_rows.append(
                    {
                        "training_seed": training_seed,
                        "method": method,
                        "domain": domain_name,
                        "frames": len(frame),
                        "scenes": frame["scene_id"].nunique(),
                        **metrics,
                    }
                )
                if method == "Full" and domain == "test":
                    bins_frame = calibration_bins(frame["y_true"], probabilities, bins)
                    bins_frame.insert(0, "training_seed", training_seed)
                    bins_frame.insert(1, "method", method)
                    calibration_rows.append(bins_frame)
                prediction_frames.append(
                    _prediction_frame(
                        frame, probabilities, training_seed, method, domain_name, "none"
                    )
                )
                scene_rows.extend(
                    _scene_metric_rows(
                        frame, probabilities, drift, training_seed, method, domain_name, bins
                    )
                )
            clean_frame = fixed_splits["test"]
            clean_probabilities, *_ = model.predict(clean_frame, "none")
            clean_f1 = classification_metrics(clean_frame["y_true"].to_numpy(int), clean_probabilities, bins)["macro_f1"]
            clean_scene_f1: dict[str, float] = {}
            for scene_id, indices in clean_frame.groupby("scene_id", sort=False).indices.items():
                idx = np.asarray(indices)
                clean_scene_f1[str(scene_id)] = classification_metrics(
                    clean_frame.iloc[idx]["y_true"].to_numpy(int), clean_probabilities[idx], bins
                )["macro_f1"]
            for pattern in cfg["study2"]["missing_patterns"]:
                probabilities, _, _, _, _ = model.predict(clean_frame, pattern)
                metrics = classification_metrics(clean_frame["y_true"].to_numpy(int), probabilities, bins)
                missing_rows.append(
                    {
                        "training_seed": training_seed,
                        "method": method,
                        "missing_pattern": pattern,
                        **metrics,
                        "macro_f1_absolute_drop": float(clean_f1 - metrics["macro_f1"]),
                        "macro_f1_relative_drop": float((clean_f1 - metrics["macro_f1"]) / max(clean_f1, 1e-12)),
                    }
                )
                if pattern != "none":
                    prediction_frames.append(
                        _prediction_frame(
                            clean_frame,
                            probabilities,
                            training_seed,
                            method,
                            "clean_controlled",
                            pattern,
                        )
                    )
                for scene_id, indices in clean_frame.groupby("scene_id", sort=False).indices.items():
                    idx = np.asarray(indices)
                    scene_metrics = classification_metrics(
                        clean_frame.iloc[idx]["y_true"].to_numpy(int), probabilities[idx], bins
                    )
                    base_f1 = clean_scene_f1[str(scene_id)]
                    missing_scene_rows.append(
                        {
                            "training_seed": training_seed,
                            "method": method,
                            "scene_id": scene_id,
                            "missing_pattern": pattern,
                            **scene_metrics,
                            "macro_f1_absolute_drop": base_f1 - scene_metrics["macro_f1"],
                            "macro_f1_relative_drop": (base_f1 - scene_metrics["macro_f1"]) / max(base_f1, 1e-12),
                        }
                    )
            # Batch-1 end-to-end CPU timing includes representation and classifier.
            timing_frame = clean_frame.groupby("scene_id", sort=False).tail(1).head(120).reset_index(drop=True)
            for _ in range(3):
                model.predict(timing_frame.iloc[:8])
            samples = []
            for index in range(len(timing_frame)):
                start = perf_counter_ns()
                model.predict(timing_frame.iloc[index : index + 1])
                samples.append((perf_counter_ns() - start) / 1.0e6)
            latency_rows.append(
                {
                    "training_seed": training_seed,
                    "method": method,
                    "batch_size": 1,
                    "p50_ms": float(np.quantile(samples, 0.50)),
                    "p95_ms": float(np.quantile(samples, 0.95)),
                    "hardware": "CPU; exact platform in run_manifest.json",
                }
            )
    latent_columns = [f"latent_{j}" for j in range(int(cfg["study2"]["latent_dim"]))]
    paired = fixed_splits["test"].merge(
        fixed_splits["stress"], on=["latent_pair_id", "frame"], suffixes=("_clean", "_stress")
    )
    latent_difference = np.max(
        np.abs(
            paired[[f"{column}_clean" for column in latent_columns]].to_numpy(float)
            - paired[[f"{column}_stress" for column in latent_columns]].to_numpy(float)
        )
    )
    return {
        "study2_seed_metrics": pd.DataFrame(result_rows),
        "study2_missing_modality": pd.DataFrame(missing_rows),
        "study2_predictions": pd.concat(prediction_frames, ignore_index=True),
        "study2_scene_metrics": pd.DataFrame(scene_rows),
        "study2_missing_scene_metrics": pd.DataFrame(missing_scene_rows),
        "study2_calibration_bins": pd.concat(calibration_rows, ignore_index=True),
        "study2_latency": pd.DataFrame(latency_rows),
        "study2_split_manifest": pd.DataFrame(split_rows),
        "study2_pair_integrity": pd.DataFrame(
            [
                {
                    "paired_clean_stress_scenes": fixed_splits["stress"]["latent_pair_id"].nunique(),
                    "paired_frames": len(paired),
                    "max_absolute_latent_difference": float(latent_difference),
                    "pass_exact_latent_pairing": bool(latent_difference == 0.0),
                }
            ]
        ),
        "study2_model_manifest": pd.DataFrame(model_rows),
    }
