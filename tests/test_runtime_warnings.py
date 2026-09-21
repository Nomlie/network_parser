from __future__ import annotations

import logging
import unittest
import warnings

from network_parser.utils import silence_expected_runtime_warnings
from network_parser.vcf_call_semantics import warn_legacy_absence_assumed_reference
from network_parser.query_engine import align_to_training_features
import pandas as pd


class TestRuntimeWarningQuiet(unittest.TestCase):
    def test_legacy_absence_logs_info_once(self):
        import network_parser.vcf_call_semantics as vcs

        vcs._LEGACY_ABSENCE_WARNING_EMITTED = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertLogs(vcs.logger, level="INFO") as logs:
                warn_legacy_absence_assumed_reference()
                warn_legacy_absence_assumed_reference()
        self.assertEqual(len(caught), 0)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("Variant-only VCF mode", logs.output[0])

    def test_partial_alignment_is_debug_not_warning(self):
        X = pd.DataFrame({"a": [1.0], "b": [1.0], "c": [1.0]}, index=["s1"])
        logger = logging.getLogger("network_parser.query_engine")
        with self.assertLogs(logger, level="DEBUG") as logs:
            logger.setLevel(logging.DEBUG)
            _, summary = align_to_training_features(X, ["a", "b", "c", "d"])
        self.assertEqual(summary["alignment_status"], "partial")
        self.assertTrue(
            any("missing" in rec.getMessage().lower() for rec in logs.records)
        )
        self.assertFalse(any(rec.levelno >= logging.WARNING for rec in logs.records))

    def test_sklearn_version_warning_is_filtered(self):
        silence_expected_runtime_warnings()
        try:
            from sklearn.exceptions import InconsistentVersionWarning
        except Exception:
            self.skipTest("sklearn InconsistentVersionWarning not available")
        categories = {
            item[2]
            for item in warnings.filters
            if item[0] == "ignore" and item[2] is not None
        }
        self.assertIn(InconsistentVersionWarning, categories)
