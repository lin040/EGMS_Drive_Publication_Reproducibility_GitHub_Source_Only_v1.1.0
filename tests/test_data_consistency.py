from __future__ import annotations

import os
import re
from pathlib import Path
import sys
import unittest

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from egms_power.statistics import two_independent_proportions_power  # noqa: E402


OUTPUT_ROOT = Path(
    os.environ.get(
        "EGMS_PUBLICATION_OUTPUT",
        REPO_ROOT / "outputs" / "publication_reproduction",
    )
).resolve()
FLOAT_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _select_one(frame: pd.DataFrame, **filters: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        mask &= frame[column].astype(str).eq(str(value))
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise AssertionError(f"Expected one row for {filters}; found {len(selected)}")
    return selected.iloc[0]


def _numbers(value: object) -> list[float]:
    return [float(match) for match in FLOAT_PATTERN.findall(str(value).replace("−", "-"))]


class DataConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not OUTPUT_ROOT.is_dir():
            raise AssertionError(
                f"Generated publication output is missing: {OUTPUT_ROOT}. "
                "Run `python run_publication.py --overwrite` first."
            )

    def assert_displayed_number(
        self,
        observed: float,
        expected: float,
        decimals: int,
    ) -> None:
        tolerance = 0.500001 * 10 ** (-decimals)
        self.assertAlmostEqual(observed, expected, delta=tolerance)

    def test_figure3_selections_and_table2_use_the_same_study2_contrasts(self) -> None:
        tables = REPO_ROOT / "data" / "studies23_frozen" / "tables"
        source = pd.read_csv(tables / "table_s2_primary_contrasts.csv")
        table2 = pd.read_csv(
            OUTPUT_ROOT / "manuscript/tables/Table_2_main_effects.csv",
            keep_default_na=False,
        )
        svg = (OUTPUT_ROOT / "manuscript/figures/Figure_3_Study2.svg").read_text(
            encoding="utf-8"
        )
        specifications = [
            ("Full vs Equal weighting", "S2 reliability"),
            ("Full vs No alignment", "S2 alignment"),
            ("Full vs No temporal consistency", "S2 temporal"),
            ("Full vs No modality dropout-distillation", "S2 missing modality"),
        ]

        self.assertEqual(set(source["comparison"]), {item[0] for item in specifications})
        for comparison, module in specifications:
            with self.subTest(comparison=comparison):
                row = _select_one(source, comparison=comparison)
                displayed = _select_one(table2, **{"Study/module": module})
                effect = float(row["difference_full_minus_ablation"])
                low = float(row["ci_low"])
                high = float(row["ci_high"])

                self.assert_displayed_number(_numbers(displayed["Effect*"])[0], effect, 4)
                displayed_ci = _numbers(displayed["95% CI"])
                self.assert_displayed_number(displayed_ci[0], low, 4)
                self.assert_displayed_number(displayed_ci[1], high, 4)

                annotation = f"Δ={effect:+.4f} [{low:+.4f}, {high:+.4f}]"
                replication = (
                    f"favorable replicates {int(row['favorable_training_replicates'])}/"
                    f"{int(row['total_training_replicates'])}; "
                    f"Holm p={float(row['holm_adjusted_p']):.4f}"
                )
                self.assertIn(annotation, svg)
                self.assertIn(replication, svg)

    def test_figure4_selections_reconcile_with_study3_contrasts_and_table2(self) -> None:
        tables = REPO_ROOT / "data" / "studies23_frozen" / "tables"
        primary = pd.read_csv(tables / "table_s3_primary_contrasts.csv")
        regimes = pd.read_csv(tables / "table_s3_by_regime.csv")
        main = pd.read_csv(tables / "table_s3_main.csv")
        controls = pd.read_csv(tables / "table_s3_negative_controls.csv")
        latency = pd.read_csv(tables / "table_s3_latency.csv")
        table2 = pd.read_csv(
            OUTPUT_ROOT / "manuscript/tables/Table_2_main_effects.csv",
            keep_default_na=False,
        )
        svg = (OUTPUT_ROOT / "manuscript/figures/Figure_4_Study3.svg").read_text(
            encoding="utf-8"
        )

        table2_specs = [
            ("Full temporal graph vs No graph", "S3 temporal graph", 1.0, 4),
            ("Full temporal graph vs Spatial-only graph", "S3 temporal relation", 1.0, 4),
            ("Full temporal graph vs No predicted-intent gating", "S3 intent gate", 1.0, 4),
            ("Full top-1 vs independently trained Single mode K=1", "S3 top-1", 100.0, 2),
        ]
        for comparison, module, scale, decimals in table2_specs:
            with self.subTest(table2_comparison=comparison):
                row = _select_one(primary, comparison=comparison)
                displayed = _select_one(table2, **{"Study/module": module})
                expected = scale * float(row["difference_full_minus_ablation"])
                low = scale * float(row["ci_low"])
                high = scale * float(row["ci_high"])
                self.assert_displayed_number(
                    _numbers(displayed["Effect*"])[0], expected, decimals
                )
                displayed_ci = _numbers(displayed["95% CI"])
                self.assert_displayed_number(displayed_ci[0], low, decimals)
                self.assert_displayed_number(displayed_ci[1], high, decimals)

        history_full = _select_one(
            regimes,
            method="Full temporal graph",
            regime="history_dependent",
            metric="minFDE_K (m)",
        )
        history_no_graph = _select_one(
            regimes,
            method="No graph",
            regime="history_dependent",
            metric="minFDE_K (m)",
        )
        history_spatial = _select_one(
            regimes,
            method="Spatial-only graph",
            regime="history_dependent",
            metric="minFDE_K (m)",
        )
        graph_contrast = _select_one(primary, comparison="Full temporal graph vs No graph")
        temporal_contrast = _select_one(
            primary,
            comparison="Full temporal graph vs Spatial-only graph",
        )
        self.assertAlmostEqual(
            float(history_full["estimate"]) - float(history_no_graph["estimate"]),
            float(graph_contrast["difference_full_minus_ablation"]),
            delta=1e-12,
        )
        self.assertAlmostEqual(
            float(history_full["estimate"]) - float(history_spatial["estimate"]),
            float(temporal_contrast["difference_full_minus_ablation"]),
            delta=1e-12,
        )

        full_brier = _select_one(main, method="Full temporal graph", metric="Brier-minFDE_6")
        no_gate_brier = _select_one(
            main,
            method="No predicted-intent gating",
            metric="Brier-minFDE_6",
        )
        gate_contrast = _select_one(
            primary,
            comparison="Full temporal graph vs No predicted-intent gating",
        )
        self.assertAlmostEqual(
            float(full_brier["estimate"]) - float(no_gate_brier["estimate"]),
            float(gate_contrast["difference_full_minus_ablation"]),
            delta=1e-12,
        )
        self.assertIn(
            f"Paired Δ={float(gate_contrast['difference_full_minus_ablation']):+.3f} "
            f"[{float(gate_contrast['ci_low']):+.3f}, "
            f"{float(gate_contrast['ci_high']):+.3f}]",
            svg,
        )

        full_top1 = _select_one(main, method="Full temporal graph", metric="MR_1 at 2 m")
        single_top1 = _select_one(main, method="Single mode K=1", metric="MR_1 at 2 m")
        top1_contrast = _select_one(
            primary,
            comparison="Full top-1 vs independently trained Single mode K=1",
        )
        self.assertAlmostEqual(
            float(full_top1["estimate"]) - float(single_top1["estimate"]),
            float(top1_contrast["difference_full_minus_ablation"]),
            delta=1e-12,
        )
        self.assertIn(
            f"Paired Δ={100 * float(top1_contrast['difference_full_minus_ablation']):+.2f} pp "
            f"[{100 * float(top1_contrast['ci_low']):+.2f}, "
            f"{100 * float(top1_contrast['ci_high']):+.2f}]",
            svg,
        )

        control_specs = [
            (
                "independent: Full vs No graph",
                "Independent: Full vs No graph; minFDE₆",
                "Independent",
            ),
            (
                "spatial: Full vs Spatial-only graph",
                "Spatial: Full vs Spatial-only; minFDE₆",
                "Spatial",
            ),
        ]
        for control_name, endpoint, figure_label in control_specs:
            with self.subTest(negative_control=control_name):
                row = _select_one(controls, negative_control=control_name)
                displayed = _select_one(table2, **{"Comparison and endpoint": endpoint})
                self.assert_displayed_number(
                    _numbers(displayed["Effect*"])[0],
                    float(row["difference_full_minus_ablation"]),
                    4,
                )
                self.assertIn(
                    f"{figure_label} Δ={float(row['difference_full_minus_ablation']):+.3f} m",
                    svg,
                )

        latency_metric = "End-to-end batch-1 CPU latency P95 (ms)"
        for method in [
            "Full temporal graph",
            "No graph",
            "Spatial-only graph",
            "No predicted-intent gating",
            "Single mode K=1",
        ]:
            with self.subTest(latency=method):
                row = _select_one(latency, method=method, metric=latency_metric)
                self.assertIn(f">{float(row['estimate']):.3f}<", svg)

    def test_figure_s1_and_table_s2_agree_at_150_450_and_749(self) -> None:
        protocol = yaml.safe_load(
            (REPO_ROOT / "configs" / "power_protocol.yaml").read_text(encoding="utf-8")
        )
        primary = protocol["primary_design"]
        p0 = float(primary["control_rate_assumption"])
        p1 = float(primary["candidate_rate_assumption"])
        alpha = float(primary["alpha"])

        detailed = pd.read_csv(OUTPUT_ROOT / "power_full/tables/table2_independent_power.csv")
        compact = pd.read_csv(
            OUTPUT_ROOT / "supplement/compact/Tables/Table_S2_planning_summary.csv",
            keep_default_na=False,
        )
        curve = pd.read_csv(OUTPUT_ROOT / "power_full/data/figure1_unpaired_power_curve.csv")
        svg = (
            OUTPUT_ROOT / "power_full/figures/figure1_unpaired_power_curve.svg"
        ).read_text(encoding="utf-8")

        power_row = _select_one(compact, Item="Power at 150 / 450 / 749")
        compact_analytic = _numbers(power_row["Value"])
        compact_mc = _numbers(power_row["Interpretation"])[:3]
        self.assertEqual(len(compact_analytic), 3)
        self.assertEqual(len(compact_mc), 3)

        for index, n in enumerate((150, 450, 749)):
            with self.subTest(episodes_per_arm=n):
                row = _select_one(detailed, episodes_per_arm=n)
                calculated = two_independent_proportions_power(n, p0, p1, alpha=alpha)
                self.assertAlmostEqual(
                    float(row["analytic_power"]),
                    calculated,
                    delta=1e-12,
                )
                self.assert_displayed_number(compact_analytic[index], calculated, 4)
                self.assert_displayed_number(compact_mc[index], float(row["mc_power"]), 4)
                self.assertIn(f"{n}: {calculated:.3f}", svg)

        for n in (150, 450):
            curve_row = _select_one(curve, episodes_per_arm=n)
            detailed_row = _select_one(detailed, episodes_per_arm=n)
            self.assertAlmostEqual(
                float(curve_row["analytic_power"]),
                float(detailed_row["analytic_power"]),
                delta=1e-12,
            )

        required = _select_one(compact, Item="Required independent n")
        required_power = _numbers(required["Interpretation"])[0]
        row_749 = _select_one(detailed, episodes_per_arm=749)
        self.assert_displayed_number(
            required_power,
            float(row_749["analytic_power"]),
            4,
        )


if __name__ == "__main__":
    unittest.main()

