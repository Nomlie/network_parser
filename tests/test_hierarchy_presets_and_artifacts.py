#!/usr/bin/env python3
"""Hierarchy presets, evaluation pack, and catalogue circularity."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from network_parser.hierarchy_artifacts import (
    assert_disjoint_sample_ids,
    resolve_global_fallback_label_columns,
    resolve_hierarchy_labels,
    write_catalogue_circularity_report,
    write_hierarchy_node_dashboard,
    write_resource_profile,
)
from network_parser.hierarchy_evaluation_pack import run_hierarchy_evaluation_pack
from network_parser.config import NetworkParserConfig


class TestHierarchyPresets(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(
            resolve_hierarchy_labels(preset="lineage_amr_profile"),
            ["Lineage_clean", "AMR_binary", "Resistance_Profile_Collapsed"],
        )
        self.assertEqual(
            resolve_hierarchy_labels(preset="lineage_family_amr_profile")[0],
            "Lineage_family",
        )

    def test_explicit_override(self):
        labels = resolve_hierarchy_labels(
            hierarchy_labels=["A", "B"],
            preset="lineage_amr_profile",
        )
        self.assertEqual(labels, ["A", "B"])


class TestGlobalFallbackLabels(unittest.TestCase):
    def test_none_by_default(self):
        cfg = NetworkParserConfig()
        cfg.hierarchy_global_fallback_labels = "none"
        labels = ["Lineage_clean", "AMR_binary", "Resistance_Profile_Collapsed"]
        self.assertEqual(resolve_global_fallback_label_columns(labels, cfg), [])

    def test_terminal_token(self):
        cfg = NetworkParserConfig()
        cfg.hierarchy_global_fallback_labels = "terminal"
        labels = ["Lineage_clean", "AMR_binary", "Resistance_Profile_Collapsed"]
        self.assertEqual(
            resolve_global_fallback_label_columns(labels, cfg),
            ["Resistance_Profile_Collapsed"],
        )

    def test_explicit_list(self):
        cfg = NetworkParserConfig()
        cfg.hierarchy_global_fallback_labels = "Lineage_clean,AMR_binary"
        labels = ["Lineage_clean", "AMR_binary", "Resistance_Profile_Collapsed"]
        self.assertEqual(
            resolve_global_fallback_label_columns(labels, cfg),
            ["Lineage_clean", "AMR_binary"],
        )

    def test_legacy(self):
        cfg = NetworkParserConfig()
        cfg.hierarchy_global_fallback_labels = "legacy"
        cfg.hierarchy_train_global_lineage_fallback = True
        labels = ["Lineage_clean", "AMR_binary"]
        got = resolve_global_fallback_label_columns(labels, cfg)
        self.assertIn("AMR_binary", got)
        self.assertIn("Lineage_clean", got)


class TestDisjointIds(unittest.TestCase):
    def test_collision_fails(self):
        with self.assertRaises(ValueError):
            assert_disjoint_sample_ids(["s1", "s2"], ["s2", "s3"])

    def test_ok(self):
        assert_disjoint_sample_ids(["s1"], ["s2"])


class TestEvaluationPack(unittest.TestCase):
    def test_pack_runs(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            pred = td / "pred.csv"
            meta = td / "meta.csv"
            pd.DataFrame(
                {
                    "sample_id": ["a", "b", "c", "d"],
                    "predicted_level1": ["L4", "L4", "L2", "L2"],
                    "predicted_level2": [
                        "susceptible",
                        "resistant",
                        "susceptible",
                        "resistant",
                    ],
                    "predicted_level3": ["Sensitive", "MDR", "Sensitive", "Mono"],
                }
            ).to_csv(pred, index=False)
            pd.DataFrame(
                {
                    "ID": ["a", "b", "c", "d"],
                    "Lineage_clean": ["lineage 4", "lineage 4", "lineage 2", "lineage 2"],
                    "AMR_binary": [
                        "susceptible",
                        "resistant",
                        "susceptible",
                        "resistant",
                    ],
                    "Resistance_Profile_Collapsed": [
                        "susceptible",
                        "MDR",
                        "susceptible",
                        "Mono",
                    ],
                }
            ).to_csv(meta, index=False)
            # Map lineage for match
            # predictions use L4/L2 short — force match by rewriting pred
            pred_df = pd.read_csv(pred)
            pred_df["predicted_level1"] = [
                "lineage 4",
                "lineage 4",
                "lineage 2",
                "lineage 2",
            ]
            pred_df.to_csv(pred, index=False)

            summary = run_hierarchy_evaluation_pack(
                predictions_path=pred,
                meta_path=meta,
                hierarchy_labels=[
                    "Lineage_clean",
                    "AMR_binary",
                    "Resistance_Profile_Collapsed",
                ],
                output_dir=td / "eval",
                harmonize_resistance_labels=True,
                n_bootstrap=50,
            )
            self.assertEqual(summary["status"], "success")
            self.assertTrue((td / "eval" / "evaluation_summary.json").exists())
            self.assertTrue((td / "eval" / "full_path_predictions.tsv").exists())


class TestCircularityAndDashboard(unittest.TestCase):
    def test_circularity(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ann = td / "ann.tsv"
            pd.DataFrame(
                {
                    "node_label": ["AMR", "AMR", "L1"],
                    "hierarchy_path": ["a", "a", "b"],
                    "catalogue_status": [
                        "known_mutation",
                        "not_in_catalogue",
                        "candidate_gene",
                    ],
                    "Feature_ID": ["f1", "f2", "f3"],
                }
            ).to_csv(ann, sep="\t", index=False)
            j = write_catalogue_circularity_report(ann, td)
            self.assertTrue(j.exists())

    def test_dashboard(self):
        reg = {
            "hierarchy": {
                "root": {
                    "level_number": 1,
                    "label_column": "Lineage_clean",
                    "status": "success",
                    "path": [],
                    "n_training_samples": 10,
                    "features": ["a", "b"],
                    "model": {"selected_algorithm": "LR"},
                    "children": {
                        "lineage 4": {
                            "level_number": 2,
                            "label_column": "AMR_binary",
                            "status": "success",
                            "path": [
                                {
                                    "level_number": "1",
                                    "label_column": "Lineage_clean",
                                    "value": "lineage 4",
                                }
                            ],
                            "n_training_samples": 5,
                            "features": ["a"],
                            "model": {"selected_algorithm": "RF"},
                            "children": {},
                        }
                    },
                }
            }
        }
        with tempfile.TemporaryDirectory() as td:
            p = write_hierarchy_node_dashboard(reg, td)
            df = pd.read_csv(p, sep="\t")
            self.assertGreaterEqual(len(df), 2)

    def test_resource_profile(self):
        with tempfile.TemporaryDirectory() as td:
            p = write_resource_profile(td, config=NetworkParserConfig(), stage="train")
            self.assertTrue(p.exists())


if __name__ == "__main__":
    unittest.main()
