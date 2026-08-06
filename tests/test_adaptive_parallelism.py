#!/usr/bin/env python3
"""Adaptive parallel worker budgeting (scales from laptop to 128GB nodes)."""

from __future__ import annotations

import unittest

from network_parser.config import NetworkParserConfig
from network_parser.utils import (
    resolve_effective_n_jobs,
    resolve_parallel_worker_budget,
    run_with_inner_n_jobs,
    should_run_parallel,
)


class TestAdaptiveParallelism(unittest.TestCase):
    def test_memory_efficient_serial_outer(self):
        cfg = NetworkParserConfig()
        cfg.n_jobs = 8
        cfg.memory_efficient = True
        budget = resolve_parallel_worker_budget(cfg, n_tasks=6)
        self.assertEqual(budget["outer_jobs"], 1)
        self.assertGreaterEqual(budget["inner_jobs"], 1)

    def test_hard_cap_respected(self):
        cfg = NetworkParserConfig()
        cfg.n_jobs = -1
        cfg.parallel_max_workers = 2
        budget = resolve_parallel_worker_budget(
            cfg, n_tasks=20, memory_per_worker_gb=0.5
        )
        self.assertLessEqual(budget["outer_jobs"], 2)

    def test_inner_context_override(self):
        cfg = NetworkParserConfig()
        cfg.n_jobs = -1

        def _probe():
            return resolve_effective_n_jobs(cfg, minimum_tasks=4)

        self.assertEqual(run_with_inner_n_jobs(3, _probe), 3)

    def test_single_task_not_parallel(self):
        cfg = NetworkParserConfig()
        self.assertFalse(
            should_run_parallel(
                cfg, enabled_attr="hierarchy_parallel_child_nodes", n_tasks=1
            )
        )

    def test_disabled_flag(self):
        cfg = NetworkParserConfig()
        cfg.hierarchy_parallel_child_nodes = False
        self.assertFalse(
            should_run_parallel(
                cfg, enabled_attr="hierarchy_parallel_child_nodes", n_tasks=10
            )
        )

    def test_outer_times_inner_reasonable(self):
        cfg = NetworkParserConfig()
        cfg.n_jobs = 12
        budget = resolve_parallel_worker_budget(
            cfg, n_tasks=6, memory_per_worker_gb=0.25
        )
        self.assertGreaterEqual(budget["outer_jobs"], 1)
        self.assertGreaterEqual(budget["inner_jobs"], 1)
        # Should not wildly oversubscribe relative to total
        self.assertLessEqual(
            budget["outer_jobs"] * budget["inner_jobs"],
            budget["total_workers"] + budget["outer_jobs"],
        )


if __name__ == "__main__":
    unittest.main()
