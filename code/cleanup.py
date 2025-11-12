#!/usr/bin/env python3
"""cleanup.py - remove generated artifacts, caches, and binaries"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

CHAPTER_PATTERNS = [
    "benchmark_results",
    "test_results",
    "test_results_*",
    "power_results",
]

PROFILE_DIR_PATTERNS = [
    "output",
    "profiling_results",
    "profiling_results_new",
    "ch*/profiling_results",
]

CACHE_DIR_PATTERNS = [
    ".pytest_cache",
]

PROFILE_FILE_PATTERNS = [
    "*.nsys-rep",
    "*.ncu-rep",
    "*.qdrep",
    "*.sqlite",
]

CACHE_FILE_PATTERNS = [
    "*.pyc",
    "*.pyo",
]

CACHE_DIR_NAMES = [
    "__pycache__",
    ".torch_inductor",
]


class CleanupRunner:
    """Main cleanup runner class."""

    def __init__(self):
        self.removed_items: List[Path] = []
        self.removed_set: Set[Path] = set()
        self.removed_by_category: Dict[str, int] = {}
        self.project_root = Path(__file__).parent.resolve()
        os.chdir(self.project_root)

    def remove_path(self, category: str, path: Path) -> None:
        """Remove a path if it exists and hasn't been removed already."""
        path = path.resolve()
        
        # Skip if already removed or doesn't exist
        if path in self.removed_set or not path.exists():
            return
        
        # Convert to relative path for display
        try:
            rel_path = path.relative_to(self.project_root)
        except ValueError:
            rel_path = path
        
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"[removed][{category}] {rel_path}")
        
        self.removed_set.add(path)
        self.removed_items.append(rel_path)
        self.removed_by_category[category] = self.removed_by_category.get(category, 0) + 1

    def remove_glob(self, category: str, pattern: str) -> None:
        """Remove files matching a glob pattern."""
        # Handle patterns with wildcards
        matches = list(self.project_root.glob(pattern))
        
        # Handle patterns in subdirectories (like ch*/profiling_results)
        if "*" in pattern or "?" in pattern:
            matches.extend(self.project_root.rglob(pattern))
        
        for match in matches:
            self.remove_path(category, match)

    def clean_chapters(self) -> None:
        """Clean chapter result folders."""
        for pattern in CHAPTER_PATTERNS:
            self.remove_glob("chapters", pattern)

    def clean_profiles(self) -> None:
        """Clean profiling captures."""
        for pattern in PROFILE_DIR_PATTERNS:
            self.remove_glob("profiles", pattern)
        
        for pattern in PROFILE_FILE_PATTERNS:
            for match in self.project_root.rglob(pattern):
                self.remove_path("profiles", match)

    def clean_caches(self) -> None:
        """Clean Python and TorchInductor caches."""
        for pattern in CACHE_DIR_PATTERNS:
            self.remove_glob("caches", pattern)
        
        # Remove cache directories
        for dir_name in CACHE_DIR_NAMES:
            for match in self.project_root.rglob(dir_name):
                if match.is_dir():
                    self.remove_path("caches", match)
        
        # Remove cache files
        for pattern in CACHE_FILE_PATTERNS:
            for match in self.project_root.rglob(pattern):
                self.remove_path("caches", match)

    def clean_binaries(self) -> None:
        """Clean built CUDA binaries."""
        # Find all .cu files and remove corresponding executables
        for cu_file in self.project_root.rglob("*.cu"):
            base = cu_file.with_suffix("")
            
            # Remove base executable if it exists and is executable
            if base.exists() and base.is_file():
                try:
                    if os.access(base, os.X_OK):
                        self.remove_path("binaries", base)
                except OSError:
                    pass
            
            # Remove variant executables (sm*, gb*)
            parent = base.parent
            base_name = base.name
            for variant_path in parent.glob(f"{base_name}_sm*"):
                if variant_path.is_file() and os.access(variant_path, os.X_OK):
                    self.remove_path("binaries", variant_path)
            for variant_path in parent.glob(f"{base_name}_gb*"):
                if variant_path.is_file() and os.access(variant_path, os.X_OK):
                    self.remove_path("binaries", variant_path)
        
        # Find ELF executables in chapter directories
        for chapter_dir in self.project_root.glob("ch*"):
            if not chapter_dir.is_dir():
                continue
            for exe in chapter_dir.iterdir():
                if not exe.is_file():
                    continue
                if not os.access(exe, os.X_OK):
                    continue
                
                # Skip if tracked by git
                try:
                    result = subprocess.run(
                        ["git", "ls-files", "--error-unmatch", str(exe)],
                        capture_output=True,
                        text=True,
                        timeout=1
                    )
                    if result.returncode == 0:
                        continue
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
                
                # Check if it's an ELF binary
                try:
                    result = subprocess.run(
                        ["file", str(exe)],
                        capture_output=True,
                        text=True,
                        timeout=1
                    )
                    if result.returncode == 0 and "ELF" in result.stdout:
                        self.remove_path("binaries", exe)
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

    def run(self) -> None:
        """Run cleanup for all categories."""
        print("Running cleanup...\n")
        
        self.clean_chapters()
        self.clean_profiles()
        self.clean_caches()
        self.clean_binaries()

    def print_summary(self) -> None:
        """Print cleanup summary."""
        print("")
        if not self.removed_items:
            print("Cleanup complete. No generated artifacts were found.")
        else:
            print(f"Cleanup complete. Removed {len(self.removed_items)} item(s).")
        
        if self.removed_by_category:
            print("")
            print("Category counts:")
            for category in ["chapters", "profiles", "caches", "binaries"]:
                count = self.removed_by_category.get(category, 0)
                if count > 0:
                    print(f"  {category:9} {count}")


def main():
    """Main entry point."""
    runner = CleanupRunner()
    runner.run()
    runner.print_summary()


if __name__ == "__main__":
    main()
