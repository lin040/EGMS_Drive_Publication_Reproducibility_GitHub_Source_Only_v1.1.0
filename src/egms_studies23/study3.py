from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from time import perf_counter_ns

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .common import TemperatureScaler, array_digest, macro_f1, softmax, stable_seed


@dataclass
class TrajectoryDataset:
    split: str
    scene_ids: np.ndarray
    regime: np.ndarray
    intent: np.ndarray
    history: np.ndarray
    neighbors: np.ndarray
    neighbor_mask: np.ndarray
    map_features: np.ndarray
    future: np.ndarray
    branch_mode: np.ndarray

    def __len__(self) -> int:
        return len(self.scene_ids)


def _integrate_velocity(velocity: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(velocity * dt, axis=0)


def _future_template(
    intent: int,
    mode: int,
    speed: float,
    future_steps: int,
    dt: float,
    interaction_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.arange(1, future_steps + 1, dtype=float) * dt
    mode_scale = (0.82, 1.0, 1.18)[mode]
    effective_speed = speed * mode_scale
    x = effective_speed * t
    y = np.zeros_like(t)
    if intent == 0:  # straight
        x += 0.10 * np.sin(0.6 * t)
    elif intent == 1:  # slowing
        deceleration = 0.55 + 0.55 * mode
        x = effective_speed * t - 0.5 * deceleration * t**2
        x = np.maximum.accumulate(np.maximum(x, 0.0))
    elif intent == 2:  # stopping
        stop_time = np.clip(2.2 + 0.55 * mode - 0.45 * interaction_strength, 1.4, 4.2)
        deceleration = effective_speed / stop_time
        x = effective_speed * np.minimum(t, stop_time) - 0.5 * deceleration * np.minimum(t, stop_time) ** 2
        x = np.maximum.accumulate(np.maximum(x, 0.0))
    elif intent == 3:  # cut in
        lane = (2.8, 3.5, 4.1)[mode]
        y = lane / (1.0 + np.exp(-(t - (2.6 - 0.35 * interaction_strength)) * 2.0))
        y -= y[0]
        x *= 0.94
    elif intent == 4:  # lane change
        direction = -1.0 if mode == 0 else 1.0
        lane = (3.2, 3.5, 3.8)[mode]
        y = direction * lane / (1.0 + np.exp(-(t - 3.0) * 1.55))
        y -= y[0]
    elif intent == 5:  # turn left
        radius = (13.0, 16.0, 19.0)[mode]
        theta = np.minimum(effective_speed * t / radius, np.pi / 2)
        x = radius * np.sin(theta)
        y = radius * (1.0 - np.cos(theta))
    elif intent == 6:  # turn right
        radius = (11.0, 14.0, 17.0)[mode]
        theta = np.minimum(effective_speed * t / radius, np.pi / 2)
        x = radius * np.sin(theta)
        y = -radius * (1.0 - np.cos(theta))
    else:  # crossing
        crossing_speed = np.clip(2.2 + 0.55 * mode, 1.5, 4.0)
        x = 0.30 * effective_speed * t
        y = crossing_speed * t * (1.0 + 0.08 * interaction_strength)
    # Smooth correlated model discrepancy prevents a tiny set of prototypes
    # from trivially reproducing every synthetic future.
    innovations = rng.normal(0.0, 0.045 + 0.012 * mode, size=(future_steps, 2))
    discrepancy = np.cumsum(innovations, axis=0)
    return np.column_stack([x, y]) + discrepancy


def generate_study3_split(
    cfg: dict, split: str, training_replicate: int | None = None
) -> TrajectoryDataset:
    study = cfg["study3"]
    n = int(study[{"train": "train_scenes", "validation": "validation_scenes", "test": "test_scenes"}[split]])
    history_steps = int(study["history_steps"])
    future_steps = int(study["future_steps"])
    dt = float(study["sample_period_s"])
    max_neighbors = int(study["max_neighbors"])
    data_seed = int(study["data_seed"])
    history = np.zeros((n, history_steps, 4), dtype=float)
    neighbors = np.zeros((n, history_steps, max_neighbors, 4), dtype=float)
    neighbor_mask = np.zeros((n, max_neighbors), dtype=bool)
    map_features = np.zeros((n, 8), dtype=float)
    future = np.zeros((n, future_steps, 2), dtype=float)
    intent = np.zeros(n, dtype=int)
    branch_mode = np.zeros(n, dtype=int)
    regime = np.empty(n, dtype=object)
    scene_ids = np.array([f"{split}_{index:05d}" for index in range(n)], dtype=object)
    for index in range(n):
        rng = np.random.default_rng(
            stable_seed(
                data_seed,
                "study3",
                split,
                training_replicate if split == "train" else "fixed",
                index,
            )
        )
        intent_index = index % 8
        intent[index] = intent_index
        mode = int(rng.integers(0, 3))
        branch_mode[index] = mode
        regime_name = ("independent", "spatial", "history_dependent")[index % 3]
        regime[index] = regime_name
        speed = float(rng.uniform(6.0, 14.0))
        history_t = (np.arange(history_steps) - history_steps + 1) * dt
        acceleration = {
            0: 0.0,
            1: -0.08,
            2: -0.12,
            3: 0.05,
            4: 0.0,
            5: -0.08,
            6: -0.08,
            7: 0.0,
        }[intent_index]
        vx = np.clip(speed + acceleration * history_t + rng.normal(0, 0.06, history_steps), 0.3, None)
        vy = rng.normal(0, 0.025, history_steps)
        # Target-only cues are intentionally incomplete for interaction-driven
        # maneuvers; this lets graph ablations test their designated mechanism.
        if intent_index in {3, 4}:
            vy += np.linspace(0.0, 0.12 if intent_index == 3 else 0.08 * (-1 if mode == 0 else 1), history_steps)
        if intent_index in {5, 6}:
            vy += np.linspace(0.0, 0.16 * (1 if intent_index == 5 else -1), history_steps)
        target_velocity = np.column_stack([vx, vy])
        target_position = _integrate_velocity(target_velocity, dt)
        target_position -= target_position[-1]
        history[index, :, :2] = target_position
        history[index, :, 2:] = target_velocity

        neighbor_count = int(rng.integers(2, max_neighbors + 1))
        neighbor_mask[index, :neighbor_count] = True
        strongest_interaction = 0.0
        for neighbor in range(neighbor_count):
            rel_x0 = float(rng.uniform(8.0, 45.0) * (-1 if rng.random() < 0.18 else 1))
            rel_y0 = float(rng.choice([-7.0, -3.5, 0.0, 3.5, 7.0]) + rng.normal(0, 0.35))
            rel_vx = float(rng.normal(-0.5, 1.7))
            rel_vy = float(rng.normal(0.0, 0.25))
            if regime_name == "independent":
                rel_x0 = float(rng.uniform(38.0, 60.0))
                rel_y0 = float(rng.choice([-10.5, -7.0, 7.0, 10.5]) + rng.normal(0, 0.3))
                rel_vx = float(rng.normal(0.0, 0.35))
                rel_vy = float(rng.normal(0.0, 0.12))
            if regime_name == "history_dependent" and neighbor > 0:
                rel_x0 = float(rng.uniform(30.0, 48.0))
                rel_y0 = float(rng.choice([-7.0, 7.0]) + rng.normal(0, 0.35))
            history_curve_x = np.zeros(history_steps)
            history_curve_y = np.zeros(history_steps)
            history_curve_vx = np.zeros(history_steps)
            history_curve_vy = np.zeros(history_steps)
            if regime_name == "history_dependent" and neighbor == 0:
                # All ambiguous maneuvers converge to similar current spatial
                # relations. Only the observed relation history identifies the
                # approach pattern; no future value is used.
                rel_x0 = 13.5 + rng.normal(0, 0.25)
                rel_y0 = 3.5 + rng.normal(0, 0.18)
                rel_vx = -0.45 + rng.normal(0, 0.08)
                rel_vy = rng.normal(0, 0.04)
                amplitude_x = {0: 0.0, 1: 4.0, 2: 7.0, 3: 2.0, 4: -2.0, 5: 1.0, 6: -1.0, 7: 5.0}[intent_index]
                amplitude_y = {0: 0.0, 1: 0.5, 2: 0.8, 3: 5.0, 4: -5.0, 5: 2.5, 6: -2.5, 7: 7.0}[intent_index]
                horizon = abs(history_t[0]) + dt
                phase = history_t / horizon
                history_curve_x = amplitude_x * phase**2
                history_curve_y = amplitude_y * phase**2
                history_curve_vx = 2.0 * amplitude_x * history_t / (horizon**2)
                history_curve_vy = 2.0 * amplitude_y * history_t / (horizon**2)
            elif regime_name == "spatial" and neighbor == 0:
                rel_x0 = 10.0 + 5.0 * (intent_index not in {1, 2, 3, 7}) + rng.normal(0, 1.1)
                rel_y0 = {3: 3.5, 4: -3.5 if mode == 0 else 3.5, 7: -8.0}.get(intent_index, rel_y0)
            rel_position = np.column_stack(
                [
                    rel_x0 + rel_vx * history_t + history_curve_x,
                    rel_y0 + rel_vy * history_t + history_curve_y,
                ]
            )
            absolute_position = target_position + rel_position
            absolute_velocity = target_velocity + np.column_stack(
                [rel_vx + history_curve_vx, rel_vy + history_curve_vy]
            )
            neighbors[index, :, neighbor, :2] = absolute_position
            neighbors[index, :, neighbor, 2:] = absolute_velocity
            closing = max(-rel_vx, 0.0)
            ttc = rel_x0 / closing if rel_x0 > 0 and closing > 0.1 else 20.0
            interaction = float(np.exp(-abs(rel_y0) / 4.0) * np.exp(-min(ttc, 20.0) / 7.0))
            strongest_interaction = max(strongest_interaction, interaction)

        intersection_probability = 0.72 if intent_index in {5, 6, 7} else 0.32
        is_intersection = float(rng.random() < intersection_probability)
        map_features[index] = np.array(
            [
                is_intersection,
                float(rng.random() < (0.68 if intent_index == 7 else 0.28)),
                float(rng.random() < (0.58 if intent_index in {3, 4} else 0.30)),
                float(rng.random() < (0.72 if intent_index == 5 else 0.34)),
                float(rng.random() < (0.72 if intent_index == 6 else 0.34)),
                float(rng.integers(1, 4)) / 3.0,
                float(rng.uniform(-0.45, 0.45)),
                float(rng.uniform(0.0, 1.0)),
            ]
        )
        interaction_strength = strongest_interaction if regime_name != "independent" else 0.0
        future[index] = _future_template(
            intent_index,
            mode,
            speed,
            future_steps,
            dt,
            interaction_strength,
            rng,
        )
    return TrajectoryDataset(
        split=split,
        scene_ids=scene_ids,
        regime=regime,
        intent=intent,
        history=history,
        neighbors=neighbors,
        neighbor_mask=neighbor_mask,
        map_features=map_features,
        future=future,
        branch_mode=branch_mode,
    )


def _linear_slope(values: np.ndarray) -> np.ndarray:
    time = np.linspace(-1.0, 0.0, values.shape[1])
    centered = time - time.mean()
    denominator = np.sum(centered**2)
    return np.sum((values - values.mean(axis=1, keepdims=True)) * centered[None, :, None], axis=1) / denominator


def graph_features(dataset: TrajectoryDataset, graph_level: str) -> np.ndarray:
    history = dataset.history
    velocity = history[:, :, 2:]
    acceleration = np.diff(velocity, axis=1) / 0.1
    speed = np.linalg.norm(velocity, axis=2)
    displacement = history[:, -1, :2] - history[:, 0, :2]
    base = np.column_stack(
        [
            velocity[:, -1, :],
            acceleration[:, -1, :],
            speed[:, -1],
            speed.mean(axis=1),
            speed.std(axis=1),
            displacement,
            _linear_slope(velocity),
            dataset.map_features,
        ]
    )
    # Current relation-aware permutation-invariant graph aggregation.
    current_target = history[:, -1, :]
    current_neighbors = dataset.neighbors[:, -1]
    rel_position = current_neighbors[:, :, :2] - current_target[:, None, :2]
    rel_velocity = current_neighbors[:, :, 2:] - current_target[:, None, 2:]
    distance = np.linalg.norm(rel_position, axis=2)
    closing = -np.sum(rel_position * rel_velocity, axis=2) / np.maximum(distance, 1.0e-6)
    ttc = np.full_like(distance, 20.0)
    np.divide(distance, closing, out=ttc, where=closing > 0.1)
    mask = dataset.neighbor_mask
    edge_mask = mask & ((distance < 25.0) | (ttc < 5.0))
    attention_logit = -distance / 18.0 - np.minimum(ttc, 20.0) / 9.0
    unnormalized = np.where(edge_mask, np.exp(attention_logit), 0.0)
    attention = unnormalized / np.maximum(unnormalized.sum(axis=1, keepdims=True), 1.0e-12)
    current_messages = np.concatenate(
        [
            rel_position,
            rel_velocity,
            distance[:, :, None],
            np.minimum(ttc, 20.0)[:, :, None],
        ],
        axis=2,
    )
    weighted = np.sum(attention[:, :, None] * current_messages, axis=1)
    min_distance = np.min(np.where(edge_mask, distance, np.inf), axis=1)
    min_ttc = np.min(np.where(edge_mask, ttc, np.inf), axis=1)
    min_distance = np.where(np.isfinite(min_distance), min_distance, 0.0)
    min_ttc = np.where(np.isfinite(min_ttc), np.minimum(min_ttc, 20.0), 0.0)
    spatial = np.column_stack(
        [weighted, min_distance, min_ttc, edge_mask.sum(axis=1)]
    )

    # Temporal graph summaries use only the observed history.
    target_position = history[:, :, None, :2]
    target_velocity = history[:, :, None, 2:]
    rel_pos_hist = dataset.neighbors[:, :, :, :2] - target_position
    rel_vel_hist = dataset.neighbors[:, :, :, 2:] - target_velocity
    distance_hist = np.linalg.norm(rel_pos_hist, axis=3)
    closing_hist = -np.sum(rel_pos_hist * rel_vel_hist, axis=3) / np.maximum(distance_hist, 1.0e-6)
    ttc_hist = np.full_like(distance_hist, 20.0)
    np.divide(distance_hist, closing_hist, out=ttc_hist, where=closing_hist > 0.1)
    nearest_distance = np.min(
        np.where(edge_mask[:, None, :], distance_hist, np.inf), axis=2
    )
    nearest_ttc = np.min(
        np.where(edge_mask[:, None, :], ttc_hist, np.inf), axis=2
    )
    nearest_distance = np.where(np.isfinite(nearest_distance), nearest_distance, 0.0)
    nearest_ttc = np.where(np.isfinite(nearest_ttc), nearest_ttc, 0.0)
    edge_count = np.maximum(edge_mask.sum(axis=1), 1)[:, None]
    mean_rel_x = np.sum(
        np.where(edge_mask[:, None, :], rel_pos_hist[:, :, :, 0], 0.0), axis=2
    ) / edge_count
    mean_rel_y = np.sum(
        np.where(edge_mask[:, None, :], rel_pos_hist[:, :, :, 1], 0.0), axis=2
    ) / edge_count
    attended_rel_x = np.sum(attention[:, None, :] * rel_pos_hist[:, :, :, 0], axis=2)
    attended_rel_y = np.sum(attention[:, None, :] * rel_pos_hist[:, :, :, 1], axis=2)
    temporal = np.column_stack(
        [
            nearest_distance[:, -1] - nearest_distance[:, 0],
            nearest_ttc[:, -1] - nearest_ttc[:, 0],
            np.min(nearest_distance, axis=1),
            np.min(nearest_ttc, axis=1),
            np.std(nearest_distance, axis=1),
            np.std(nearest_ttc, axis=1),
            _linear_slope(nearest_distance[:, :, None])[:, 0],
            _linear_slope(nearest_ttc[:, :, None])[:, 0],
            mean_rel_x[:, -1] - mean_rel_x[:, 0],
            mean_rel_y[:, -1] - mean_rel_y[:, 0],
            _linear_slope(mean_rel_x[:, :, None])[:, 0],
            _linear_slope(mean_rel_y[:, :, None])[:, 0],
            attended_rel_x[:, -1] - attended_rel_x[:, 0],
            attended_rel_y[:, -1] - attended_rel_y[:, 0],
            _linear_slope(attended_rel_x[:, :, None])[:, 0],
            _linear_slope(attended_rel_y[:, :, None])[:, 0],
        ]
    )
    full = np.concatenate([base, spatial, temporal], axis=1)
    if graph_level == "full":
        return full
    if graph_level == "spatial":
        return np.concatenate([base, spatial, np.zeros_like(temporal)], axis=1)
    if graph_level == "none":
        return np.concatenate([base, np.zeros_like(spatial), np.zeros_like(temporal)], axis=1)
    raise ValueError(f"Unknown graph level: {graph_level}")


@dataclass(frozen=True)
class Study3Condition:
    name: str
    graph_level: str = "full"
    predicted_intent_gating: bool = True
    modes: int = 6


CONDITIONS = {
    "Full temporal graph": Study3Condition("Full temporal graph"),
    "No graph": Study3Condition("No graph", graph_level="none"),
    "Spatial-only graph": Study3Condition("Spatial-only graph", graph_level="spatial"),
    "No predicted-intent gating": Study3Condition(
        "No predicted-intent gating", graph_level="full", predicted_intent_gating=False
    ),
    "Single mode K=1": Study3Condition("Single mode K=1", graph_level="full", modes=1),
}


class Study3Model:
    def __init__(self, cfg: dict, condition: Study3Condition, training_seed: int):
        self.cfg = cfg
        self.condition = condition
        self.training_seed = int(training_seed)
        study = cfg["study3"]
        self.intent_scaler = StandardScaler()
        self.intent_classifier = LogisticRegression(
            C=float(study["classifier_C"]),
            max_iter=int(study["classifier_max_iter"]),
            class_weight="balanced",
            random_state=training_seed,
            solver="lbfgs",
        )
        self.intent_temperature = TemperatureScaler()
        self.intent_prototypes: dict[int, np.ndarray] = {}
        self.intent_priors: dict[int, np.ndarray] = {}
        self.training_intent_prior = np.full(
            len(self.cfg["study3"]["intent_classes"]),
            1.0 / len(self.cfg["study3"]["intent_classes"]),
        )

    @staticmethod
    def _speed_scale(dataset: TrajectoryDataset) -> np.ndarray:
        return np.maximum(np.linalg.norm(dataset.history[:, -1, 2:], axis=1), 2.0)

    def fit(self, training: TrajectoryDataset, validation: TrajectoryDataset) -> "Study3Model":
        x_train = graph_features(training, self.condition.graph_level)
        x_validation = graph_features(validation, self.condition.graph_level)
        self.intent_scaler.fit(x_train)
        self.intent_classifier.fit(self.intent_scaler.transform(x_train), training.intent)
        validation_raw = self.intent_classifier.predict_proba(self.intent_scaler.transform(x_validation))
        self.intent_temperature.fit(validation_raw, validation.intent)
        normalized_future = training.future / self._speed_scale(training)[:, None, None]
        downsample = normalized_future[:, 4::5, :].reshape(len(training), -1)
        prototypes_per_intent = int(self.cfg["study3"]["prototypes_per_intent"])
        counts = np.bincount(
            training.intent,
            minlength=len(self.cfg["study3"]["intent_classes"]),
        ).astype(float)
        self.training_intent_prior = counts / counts.sum()
        cluster_count = 1 if self.condition.modes == 1 else prototypes_per_intent
        for intent_index in range(len(self.cfg["study3"]["intent_classes"])):
            mask = training.intent == intent_index
            kmeans = KMeans(
                n_clusters=cluster_count,
                random_state=stable_seed(self.training_seed, "intent_cluster", intent_index),
                n_init=10,
            ).fit(downsample[mask])
            labels = kmeans.labels_
            prototypes = []
            priors = []
            for cluster in range(cluster_count):
                members = normalized_future[mask][labels == cluster]
                prototypes.append(members.mean(axis=0))
                priors.append(len(members))
            self.intent_prototypes[intent_index] = np.stack(prototypes)
            self.intent_priors[intent_index] = np.asarray(priors, dtype=float) / np.sum(priors)
        return self

    def intent_probabilities(self, dataset: TrajectoryDataset) -> np.ndarray:
        features = graph_features(dataset, self.condition.graph_level)
        raw = self.intent_classifier.predict_proba(self.intent_scaler.transform(features))
        return self.intent_temperature.transform(raw)

    def predict(self, dataset: TrajectoryDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        intent_probabilities = self.intent_probabilities(dataset)
        n = len(dataset)
        output_k = int(self.condition.modes)
        future_steps = int(self.cfg["study3"]["future_steps"])
        candidates = np.zeros((n, output_k, future_steps, 2), dtype=float)
        probabilities = np.zeros((n, output_k), dtype=float)
        scales = self._speed_scale(dataset)
        gating = (
            intent_probabilities
            if self.condition.predicted_intent_gating
            else np.broadcast_to(self.training_intent_prior, intent_probabilities.shape)
        )
        for row in range(n):
            options = []
            for intent_index, prototypes in self.intent_prototypes.items():
                for mode_index, prototype in enumerate(prototypes):
                    probability = gating[row, intent_index] * self.intent_priors[intent_index][mode_index]
                    options.append((float(probability), intent_index, mode_index, prototype))
            options.sort(key=lambda item: (-item[0], item[1], item[2]))
            selected = options[:output_k]
            probabilities[row] = np.array([item[0] for item in selected])
            probabilities[row] /= max(probabilities[row].sum(), 1.0e-12)
            candidates[row] = np.stack([item[3] * scales[row] for item in selected])
        return candidates, probabilities, intent_probabilities


def trajectory_scene_metrics(
    ground_truth: np.ndarray,
    candidates: np.ndarray,
    probabilities: np.ndarray,
    miss_threshold: float,
) -> dict[str, np.ndarray]:
    errors = np.linalg.norm(candidates - ground_truth[:, None, :, :], axis=3)
    ade_by_mode = errors.mean(axis=2)
    fde_by_mode = errors[:, :, -1]
    best_fde_mode = fde_by_mode.argmin(axis=1)
    row = np.arange(len(ground_truth))
    # Official AV2 aggregation selects k* by minimum endpoint error and then
    # uses that same mode for minADE and Brier-minFDE.
    min_ade = ade_by_mode[row, best_fde_mode]
    min_fde = fde_by_mode[row, best_fde_mode]
    miss = (min_fde > miss_threshold).astype(float)
    brier_min_fde = min_fde + (1.0 - probabilities[row, best_fde_mode]) ** 2
    top = probabilities.argmax(axis=1)
    top1_ade = ade_by_mode[np.arange(len(ground_truth)), top]
    top1_fde = fde_by_mode[np.arange(len(ground_truth)), top]
    top1_miss = (top1_fde > miss_threshold).astype(float)
    return {
        "minADE_K": min_ade,
        "minFDE_K": min_fde,
        "MR_K": miss,
        "Brier-minFDE_K": brier_min_fde,
        "ADE_1": top1_ade,
        "FDE_1": top1_fde,
        "MR_1": top1_miss,
    }


def trajectory_mode_errors(
    ground_truth: np.ndarray, candidates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    errors = np.linalg.norm(candidates - ground_truth[:, None, :, :], axis=3)
    return errors.mean(axis=2), errors[:, :, -1]


def _trajectory_dataset_digest(dataset: TrajectoryDataset) -> str:
    text = "\x1f".join(
        [
            dataset.split,
            *map(str, dataset.scene_ids),
            *map(str, dataset.regime),
        ]
    )
    return array_digest(
        np.frombuffer(text.encode("utf-8"), dtype=np.uint8),
        dataset.intent,
        dataset.history,
        dataset.neighbors,
        dataset.neighbor_mask.astype(np.uint8),
        dataset.map_features,
        dataset.future,
        dataset.branch_mode,
    )


def _study3_parameter_arrays(model: Study3Model) -> list[np.ndarray]:
    arrays: list[np.ndarray] = [
        model.intent_scaler.mean_,
        model.intent_scaler.scale_,
        model.intent_classifier.coef_,
        model.intent_classifier.intercept_,
        model.intent_classifier.classes_,
        np.array([model.intent_temperature.temperature]),
        model.training_intent_prior,
    ]
    for intent_index in sorted(model.intent_prototypes):
        arrays.extend(
            [model.intent_prototypes[intent_index], model.intent_priors[intent_index]]
        )
    return arrays


def run_study3(cfg: dict) -> dict[str, pd.DataFrame | np.ndarray]:
    fixed_splits = {
        split: generate_study3_split(cfg, split)
        for split in ("validation", "test")
    }
    methods = list(cfg["study3"]["methods"])
    seeds = list(cfg["common"]["training_seeds"])
    prediction_rows = []
    seed_rows = []
    latency_rows = []
    model_rows = []
    split_rows = []
    example_payload = None
    for name, data in fixed_splits.items():
        split_rows.append(
            {
                "split": name,
                "training_seed": np.nan,
                "scenes": len(data),
                "intent_classes": len(np.unique(data.intent)),
                "regimes": ",".join(sorted(np.unique(data.regime))),
                "history_steps": data.history.shape[1],
                "future_steps": data.future.shape[1],
                "agent_history_future_overlap": 0,
                "data_sha256": _trajectory_dataset_digest(data),
            }
        )
    for training_seed in seeds:
        training = generate_study3_split(cfg, "train", int(training_seed))
        training_hash = _trajectory_dataset_digest(training)
        split_rows.append(
            {
                "split": "train",
                "training_seed": training_seed,
                "scenes": len(training),
                "intent_classes": len(np.unique(training.intent)),
                "regimes": ",".join(sorted(np.unique(training.regime))),
                "history_steps": training.history.shape[1],
                "future_steps": training.future.shape[1],
                "agent_history_future_overlap": 0,
                "data_sha256": training_hash,
            }
        )
        for method in methods:
            model = Study3Model(cfg, CONDITIONS[method], int(training_seed)).fit(
                training, fixed_splits["validation"]
            )
            parameter_arrays = _study3_parameter_arrays(model)
            inference_array = np.frombuffer(
                json.dumps(asdict(model.condition), sort_keys=True).encode("utf-8"),
                dtype=np.uint8,
            )
            model_rows.append(
                {
                    "training_seed": training_seed,
                    "method": method,
                    "model_sha256": array_digest(*parameter_arrays, inference_array),
                    "parameter_sha256": array_digest(*parameter_arrays),
                    "inference_rule_sha256": array_digest(inference_array),
                    "training_data_sha256": training_hash,
                    "numeric_state_values": int(sum(array.size for array in parameter_arrays)),
                    "hash_scope": "complete numeric fitted state + inference condition",
                    "temperature": float(model.intent_temperature.temperature),
                    "graph_level": model.condition.graph_level,
                    "predicted_intent_gating": model.condition.predicted_intent_gating,
                    "K": model.condition.modes,
                }
            )
            candidates, probabilities, intent_probabilities = model.predict(fixed_splits["test"])
            metrics = trajectory_scene_metrics(
                fixed_splits["test"].future,
                candidates,
                probabilities,
                float(cfg["study3"]["miss_threshold_m"]),
            )
            ade_by_mode, fde_by_mode = trajectory_mode_errors(
                fixed_splits["test"].future, candidates
            )
            intent_prediction = intent_probabilities.argmax(axis=1)
            seed_row = {
                "training_seed": training_seed,
                "method": method,
                "K": int(CONDITIONS[method].modes),
                "intent_macro_f1": macro_f1(fixed_splits["test"].intent, intent_prediction, 8),
                **{name: float(values.mean()) for name, values in metrics.items()},
            }
            if int(CONDITIONS[method].modes) == 1:
                # Never mislabel K=1 as K=6 in manuscript tables.
                seed_row["minADE_6_reportable"] = np.nan
                seed_row["minFDE_6_reportable"] = np.nan
                seed_row["MR_6_reportable"] = np.nan
                seed_row["Brier-minFDE_6_reportable"] = np.nan
            else:
                seed_row["minADE_6_reportable"] = seed_row["minADE_K"]
                seed_row["minFDE_6_reportable"] = seed_row["minFDE_K"]
                seed_row["MR_6_reportable"] = seed_row["MR_K"]
                seed_row["Brier-minFDE_6_reportable"] = seed_row["Brier-minFDE_K"]
            seed_rows.append(seed_row)
            max_modes = int(cfg["study3"]["modes"])
            for row in range(len(fixed_splits["test"])):
                record = {
                        "training_seed": training_seed,
                        "method": method,
                        "scene_id": fixed_splits["test"].scene_ids[row],
                        "regime": fixed_splits["test"].regime[row],
                        "K": int(CONDITIONS[method].modes),
                        "intent_true": int(fixed_splits["test"].intent[row]),
                        "intent_pred": int(intent_prediction[row]),
                        **{name: float(values[row]) for name, values in metrics.items()},
                    }
                for intent_index in range(intent_probabilities.shape[1]):
                    record[f"intent_p_{intent_index}"] = float(intent_probabilities[row, intent_index])
                for mode_index in range(max_modes):
                    active = mode_index < probabilities.shape[1]
                    record[f"p_mode_{mode_index + 1}"] = float(probabilities[row, mode_index]) if active else np.nan
                    record[f"ade_mode_{mode_index + 1}"] = float(ade_by_mode[row, mode_index]) if active else np.nan
                    record[f"fde_mode_{mode_index + 1}"] = float(fde_by_mode[row, mode_index]) if active else np.nan
                record["best_fde_mode"] = int(np.argmin(fde_by_mode[row]) + 1)
                prediction_rows.append(record)
            timing_indices = np.arange(min(120, len(fixed_splits["test"])))
            samples = []
            for index in timing_indices:
                single = subset_dataset(fixed_splits["test"], np.array([index]))
                start = perf_counter_ns()
                model.predict(single)
                samples.append((perf_counter_ns() - start) / 1.0e6)
            latency_rows.append(
                {
                    "training_seed": training_seed,
                    "method": method,
                    "batch_size": 1,
                    "p50_ms": float(np.quantile(samples, 0.50)),
                    "p95_ms": float(np.quantile(samples, 0.95)),
                    "includes": "feature extraction + graph aggregation + intent + trajectory decoding",
                }
            )
            if training_seed == seeds[0] and method == "Full temporal graph":
                example_indices = np.array([3, 19, 41, 77])
                example_payload = {
                    "scene_ids": fixed_splits["test"].scene_ids[example_indices],
                    "history": fixed_splits["test"].history[example_indices, :, :2],
                    "future": fixed_splits["test"].future[example_indices],
                    "candidates": candidates[example_indices],
                    "probabilities": probabilities[example_indices],
                    "intent": fixed_splits["test"].intent[example_indices],
                }
    assert example_payload is not None
    return {
        "study3_seed_metrics": pd.DataFrame(seed_rows),
        "study3_predictions": pd.DataFrame(prediction_rows),
        "study3_latency": pd.DataFrame(latency_rows),
        "study3_split_manifest": pd.DataFrame(split_rows),
        "study3_model_manifest": pd.DataFrame(model_rows),
        "study3_examples": example_payload,
    }


def subset_dataset(dataset: TrajectoryDataset, indices: np.ndarray) -> TrajectoryDataset:
    return TrajectoryDataset(
        split=dataset.split,
        scene_ids=dataset.scene_ids[indices],
        regime=dataset.regime[indices],
        intent=dataset.intent[indices],
        history=dataset.history[indices],
        neighbors=dataset.neighbors[indices],
        neighbor_mask=dataset.neighbor_mask[indices],
        map_features=dataset.map_features[indices],
        future=dataset.future[indices],
        branch_mode=dataset.branch_mode[indices],
    )
