"""Legacy shim for the UMA memory lab benchmarks.

Use `labs/uma_memory/uma_memory_reporting_optimized_benchmark.py` instead.
"""

from __future__ import annotations

from labs.uma_memory import uma_memory_reporting_optimized_benchmark as _impl


def get_benchmark():
    return _impl.get_benchmark()
