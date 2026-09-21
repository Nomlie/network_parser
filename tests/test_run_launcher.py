#!/usr/bin/env python3
"""User-facing run_network_parser.py launcher and preflight checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from network_parser.preflight import (
    format_preflight_report,
    load_and_validate_config,
    min_train_vcf_threshold,
    validate_evaluate,
    validate_query,
    validate_train_hierarchy,
)

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_network_parser.py"


def _run_launcher(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(LAUNCHER), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _tiny_vcf(path: Path, sample: str) -> Path:
    _write(
        path,
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + sample
        + "\n",
    )
    return path


def _train_namespace(**overrides) -> argparse.Namespace:
    values = dict(
        command="train-hierarchy",
        genomic=None,
        meta=None,
        output_dir="out",
        config=None,
        ref_fasta=None,
        hierarchy_labels=None,
        hierarchy_preset=None,
        level1_label=None,
        level2_label=None,
        level2_binary_label_mapping_file=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _query_namespace(**overrides) -> argparse.Namespace:
    values = dict(
        command="query",
        genomic=None,
        output_dir="out",
        config=None,
        ref_fasta=None,
        bundle=None,
        registry=None,
        query_input_type="auto",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class TestLauncherHelp(unittest.TestCase):
    def test_root_help_describes_two_modes(self):
        result = _run_launcher("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("train-hierarchy", result.stdout)
        self.assertIn("query", result.stdout)
        self.assertIn(
            "python run_network_parser.py train-hierarchy --help", result.stdout
        )
        self.assertIn("python run_network_parser.py query --help", result.stdout)
        self.assertIn("python run_network_parser.py evaluate --help", result.stdout)
        self.assertIn("evaluate", result.stdout)
        self.assertNotIn("--rf_selector_n_estimators", result.stdout)
        self.assertNotIn("--hierarchy_labels", result.stdout.split("Examples:")[0])

    def test_no_args_shows_general_help(self):
        result = _run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Main commands:", result.stdout)
        self.assertIn("Other commands:", result.stdout)

    def test_train_help_shows_training_options(self):
        result = _run_launcher("train-hierarchy", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--genomic", result.stdout)
        self.assertIn("--meta", result.stdout)
        self.assertIn("--hierarchy_labels", result.stdout)
        self.assertIn("--output_dir", result.stdout)

    def test_query_help_shows_query_options(self):
        result = _run_launcher("query", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--genomic", result.stdout)
        self.assertIn("--bundle", result.stdout)
        self.assertIn("--registry", result.stdout)
        self.assertIn("--query_input_type", result.stdout)

    def test_evaluate_help_shows_evaluate_options(self):
        result = _run_launcher("evaluate", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--predictions", result.stdout)
        self.assertIn("--meta", result.stdout)
        self.assertIn("--label", result.stdout)

    def test_other_commands_have_help(self):
        for command in (
            "run",
            "bundle",
            "evaluate-hierarchy",
            "cross-validate",
            "annotate-panels",
            "train-two-level",
        ):
            result = _run_launcher(command, "--help")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("-h, --help", result.stdout)

    def test_unknown_command_exits_cleanly(self):
        result = _run_launcher("not-a-mode")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown command", result.stderr)
        self.assertIn("--help", result.stderr)

    def test_missing_required_args_are_short(self):
        result = _run_launcher("train-hierarchy")
        self.assertEqual(result.returncode, 2)
        self.assertIn("required", result.stderr)
        self.assertIn("--help", result.stderr)
        self.assertLess(len(result.stderr.splitlines()), 8)


class TestConfigPreflight(unittest.TestCase):
    def test_missing_config(self):
        _, errors = load_and_validate_config("/tmp/networkparser-missing-config.json")
        self.assertTrue(any("not found" in item for item in errors))

    def test_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "config.json", "{not json")
            _, errors = load_and_validate_config(str(path))
        self.assertTrue(any("not valid JSON" in item for item in errors))

    def test_unknown_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(
                Path(tmp) / "config.json",
                json.dumps({"not_a_real_setting": 1, "n_jobs": 2}),
            )
            _, errors = load_and_validate_config(str(path))
        self.assertTrue(any("unknown setting" in item for item in errors))

    def test_invalid_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(
                Path(tmp) / "config.json",
                json.dumps({"ancestral_allele": "Z"}),
            )
            _, errors = load_and_validate_config(str(path))
        self.assertTrue(any("invalid setting" in item for item in errors))

    def test_valid_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(
                Path(tmp) / "config.json",
                json.dumps({"n_jobs": 2, "min_sample_presence": 4}),
            )
            config, errors = load_and_validate_config(str(path))
        self.assertEqual(errors, [])
        self.assertEqual(config.n_jobs, 2)
        self.assertEqual(min_train_vcf_threshold(config), 4)


class TestTrainPreflight(unittest.TestCase):
    def test_missing_genomic_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = _write(
                Path(tmp) / "meta.csv",
                "ID,Lineage,AMR_binary\ns1,L1,R\n",
            )
            args = _train_namespace(
                genomic=str(Path(tmp) / "missing_vcfs"),
                meta=str(meta),
                hierarchy_labels=["Lineage", "AMR_binary"],
            )
            errors = validate_train_hierarchy(args)
        self.assertTrue(any("not found" in item for item in errors))

    def test_too_few_vcf_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genomic = root / "vcfs"
            genomic.mkdir()
            _tiny_vcf(genomic / "s1.vcf", "s1")
            _tiny_vcf(genomic / "s2.vcf", "s2")
            meta = _write(
                root / "meta.csv",
                "ID,Lineage,AMR_binary,vcf_file\n"
                "s1,L1,R,s1.vcf\n"
                "s2,L2,S,s2.vcf\n",
            )
            args = _train_namespace(
                genomic=str(genomic),
                meta=str(meta),
                hierarchy_labels=["Lineage", "AMR_binary"],
            )
            errors = validate_train_hierarchy(args)
        joined = " ".join(errors)
        self.assertIn("at least", joined)
        self.assertIn("VCF", joined)

    def test_missing_label_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genomic = root / "vcfs"
            genomic.mkdir()
            for i in range(10):
                _tiny_vcf(genomic / f"s{i}.vcf", f"s{i}")
            meta = _write(
                root / "meta.csv",
                "ID,Lineage\n" + "".join(f"s{i},L1\n" for i in range(10)),
            )
            args = _train_namespace(
                genomic=str(genomic),
                meta=str(meta),
                hierarchy_labels=["Lineage", "AMR_binary"],
            )
            errors = validate_train_hierarchy(args)
        self.assertTrue(any("AMR_binary" in item for item in errors))

    def test_valid_training_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genomic = root / "vcfs"
            genomic.mkdir()
            rows = ["ID,Lineage,AMR_binary,vcf_file"]
            for i in range(10):
                _tiny_vcf(genomic / f"s{i}.vcf", f"s{i}")
                rows.append(f"s{i},L1,R,s{i}.vcf")
            meta = _write(root / "meta.csv", "\n".join(rows) + "\n")
            args = _train_namespace(
                genomic=str(genomic),
                meta=str(meta),
                hierarchy_labels=["Lineage", "AMR_binary"],
                output_dir=str(root / "out"),
            )
            self.assertEqual(validate_train_hierarchy(args), [])

    def test_demo_train_inputs_pass_preflight(self):
        args = _train_namespace(
            genomic=str(ROOT / "data" / "train"),
            meta=str(ROOT / "data" / "train_metadata.csv"),
            hierarchy_labels=["Lineage_clean", "AMR_binary"],
            ref_fasta=str(ROOT / "data" / "reference" / "H37Rv.fasta"),
            output_dir=str(ROOT / "demo_results" / "train"),
        )
        self.assertEqual(validate_train_hierarchy(args), [])


class TestQueryPreflight(unittest.TestCase):
    def test_requires_bundle_or_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            genomic = Path(tmp) / "vcfs"
            genomic.mkdir()
            _tiny_vcf(genomic / "s1.vcf", "s1")
            args = _query_namespace(
                genomic=str(genomic), output_dir=str(Path(tmp) / "out")
            )
            errors = validate_query(args)
        self.assertTrue(
            any("exactly one trained model source" in item for item in errors)
        )

    def test_missing_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            genomic = Path(tmp) / "vcfs"
            genomic.mkdir()
            _tiny_vcf(genomic / "s1.vcf", "s1")
            args = _query_namespace(
                genomic=str(genomic),
                bundle=str(Path(tmp) / "missing.npb"),
                output_dir=str(Path(tmp) / "out"),
            )
            errors = validate_query(args)
        self.assertTrue(any("Model bundle not found" in item for item in errors))

    def test_valid_query_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genomic = root / "vcfs"
            genomic.mkdir()
            _tiny_vcf(genomic / "s1.vcf", "s1")
            bundle = _write(root / "model.npb", "bundle-placeholder\n")
            args = _query_namespace(
                genomic=str(genomic),
                bundle=str(bundle),
                output_dir=str(root / "out"),
            )
            self.assertEqual(validate_query(args), [])

    def test_launcher_reports_preflight_and_exits(self):
        result = _run_launcher(
            "train-hierarchy",
            "--genomic",
            "/definitely/missing/vcfs",
            "--meta",
            "/definitely/missing/meta.csv",
            "--hierarchy_labels",
            "Lineage",
            "AMR_binary",
            "--output_dir",
            "/tmp/networkparser-preflight-out",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("NetworkParser could not start", result.stderr)
        self.assertIn("train-hierarchy --help", result.stderr)

    def test_report_format(self):
        text = format_preflight_report(
            ["first problem", "second problem"], command="query"
        )
        self.assertIn("1. first problem", text)
        self.assertIn("2. second problem", text)
        self.assertIn("query --help", text)


class TestEvaluatePreflight(unittest.TestCase):
    def test_missing_predictions_and_meta(self):
        args = argparse.Namespace(
            command="evaluate",
            predictions="/no/such/predictions.csv",
            meta="/no/such/meta.csv",
            output_dir="/tmp/eval-out",
            label="AMR_binary",
            hierarchy_labels=None,
        )
        errors = validate_evaluate(args)
        joined = " ".join(errors)
        self.assertIn("Predictions file not found", joined)
        self.assertIn("Metadata file not found", joined)

    def test_launcher_evaluate_missing_files(self):
        result = _run_launcher(
            "evaluate",
            "--predictions",
            "/no/such/predictions.csv",
            "--meta",
            "/no/such/meta.csv",
            "--label",
            "AMR_binary",
            "--output_dir",
            "/tmp/networkparser-eval-out",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("NetworkParser could not start", result.stderr)
        self.assertIn("evaluate --help", result.stderr)


if __name__ == "__main__":
    unittest.main()
