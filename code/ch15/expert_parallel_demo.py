"""expert_parallel_demo.py - Chapter 15 expert-parallel (EP) demo (tool).

Alias wrapper around `ch15/expert_parallelism.py` so the chapter has a clear
`*_demo.py` entrypoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch15.expert_parallelism import main


if __name__ == "__main__":
    main()

