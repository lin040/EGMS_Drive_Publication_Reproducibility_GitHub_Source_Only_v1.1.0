from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from egms_study1r import generator as compatibility_generator  # noqa: E402
from egms_study1r2 import generator as frozen_generator  # noqa: E402
from egms_study1r2.runner import _source_hashes  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RevisedStudy1Tests(unittest.TestCase):
    def test_frozen_final_source_hashes_are_exact(self) -> None:
        freeze = json.loads(
            (ROOT / "configs" / "study1r2_freeze_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(_source_hashes(ROOT), freeze["final_source_sha256"])
        self.assertFalse(freeze["validation_rule_requires_improvement"])
        self.assertTrue(freeze["post_hoc_after_study1r"])
        self.assertTrue(freeze["not_preregistered"])

    def test_method_blind_generator_is_preserved(self) -> None:
        self.assertEqual(
            digest(ROOT / "src/egms_study1r/generator.py"),
            digest(ROOT / "src/egms_study1r2/generator.py"),
        )
        for function in (
            compatibility_generator.make_scenario_tape,
            compatibility_generator.generate_offline_split,
            compatibility_generator.observe_state,
            compatibility_generator.oracle_actions,
            compatibility_generator.step_dynamics,
        ):
            parameters = " ".join(inspect.signature(function).parameters).lower()
            self.assertNotIn("method", parameters)
            self.assertNotIn("baseline", parameters)
            self.assertNotIn("structured", parameters)

    def test_public_inputs_have_one_two_arm_naming_scheme(self) -> None:
        data = ROOT / "data" / "study1_frozen"
        summary = pd.read_csv(data / "study1_metric_summary.csv")
        contrasts = pd.read_csv(data / "study1_paired_contrasts.csv")
        replicates = pd.read_csv(data / "study1_replicate_effects.csv")
        self.assertEqual(set(summary["method"]), {"Baseline B", "Structured fusion"})
        self.assertEqual(set(contrasts["display_contrast"]), {"Structured fusion - Baseline B"})
        self.assertIn("structured_fusion_estimate", replicates.columns)
        self.assertNotIn("structured_estimate", replicates.columns)
        for path in (
            data / "study1_metric_summary.csv",
            data / "study1_paired_contrasts.csv",
            data / "study1_replicate_effects.csv",
        ):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("Structured-R2", text)
            self.assertNotIn("Study 1-R2", text)

    def test_all_ten_endpoints_and_directionality_are_retained(self) -> None:
        contrasts = pd.read_csv(ROOT / "data/study1_frozen/study1_paired_contrasts.csv")
        self.assertEqual(
            set(contrasts["endpoint"]),
            {
                "macro_f1", "nll", "brier", "ece", "collision", "near_miss",
                "critical_event", "route_completion", "ttc_p5", "jerk_p95",
            },
        )
        total = (
            contrasts["favorable_replicates"]
            + contrasts["tied_replicates"]
            + contrasts["adverse_replicates"]
        )
        self.assertTrue(total.eq(10).all())
        self.assertTrue(contrasts["validator_requires_support"].eq(False).all())
        supported = set(contrasts.loc[contrasts["support_rule_met"], "endpoint"])
        self.assertEqual(supported, {"macro_f1", "nll", "brier"})

    def test_revised_figure2_is_exact_manuscript_asset(self) -> None:
        asset = (
            ROOT
            / "publication_assets/manuscript/Figure_2_Study1_Baseline_B_vs_Structured_fusion.png"
        )
        self.assertEqual(
            digest(asset),
            "786e10a5fc734d8848ea5d19ffca4dd51dc1e7ef65ccb8131c5a67445f4dcfb9",
        )
        svg = asset.with_suffix(".svg").read_text(encoding="utf-8")
        self.assertIn("Baseline B", svg)
        self.assertIn("Structured fusion", svg)
        self.assertNotIn("Structured-R2", svg)
        self.assertNotIn("Study 1-R2", svg)

    def test_public_provenance_hashes_match(self) -> None:
        data = ROOT / "data" / "study1_frozen"
        provenance = json.loads((data / "PROVENANCE.json").read_text(encoding="utf-8"))
        for relative, expected in provenance["canonical_publication_inputs"].items():
            self.assertEqual(digest(data / relative), expected, relative)
        assets = ROOT / "publication_assets" / "manuscript"
        for relative, expected in provenance["expected_publication_assets"].items():
            self.assertEqual(digest(assets / relative), expected, relative)
        self.assertTrue(provenance["validation"]["passed"])
        self.assertTrue(provenance["validation"]["direction_neutral"])
        self.assertFalse(provenance["validation"]["requires_structured_improvement"])


if __name__ == "__main__":
    unittest.main()
