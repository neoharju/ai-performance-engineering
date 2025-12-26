"""context_parallel_demo.py - Chapter 15 context-parallel (CP) demo (tool).

This wrapper calls the Chapter 13 context-parallelism implementation, which is
shared across chapters.

Run with torchrun, e.g.:
  torchrun --nproc_per_node <num_gpus> ch15/context_parallel_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch13.context_parallelism import main


if __name__ == "__main__":
    main()
