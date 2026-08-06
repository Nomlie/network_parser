#!/usr/bin/env python3
# network_parser/two_level_protocol.py
"""
Backward-compatible import shim for the hierarchy protocol.

Prefer::

    from network_parser.hierarchy_protocol import HierarchyProtocol

Legacy imports still work::

    from network_parser.two_level_protocol import TwoLevelProtocol
"""

from __future__ import annotations

try:
    from network_parser.hierarchy_protocol import HierarchyProtocol
except ImportError:  # pragma: no cover - source-tree execution
    from hierarchy_protocol import HierarchyProtocol  # type: ignore

TwoLevelProtocol: type[HierarchyProtocol] = HierarchyProtocol

__all__ = ["HierarchyProtocol", "TwoLevelProtocol"]
