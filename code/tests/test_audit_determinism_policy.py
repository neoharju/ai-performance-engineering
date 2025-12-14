from __future__ import annotations

from pathlib import Path

from core.scripts.audit_verification_compliance import audit_directory


def _write_benchmark(tmp_path: Path, filename: str, source: str) -> Path:
    path = tmp_path / filename
    path.write_text(source, encoding="utf-8")
    return path


def _get_single_result(results: dict) -> dict:
    assert len(results) == 1, f"Expected exactly 1 audited file, got {len(results)}"
    return next(iter(results.values()))


def test_audit_fails_when_determinism_enabled_without_justification(tmp_path: Path) -> None:
    _write_benchmark(
        tmp_path,
        "baseline_determinism_on.py",
        (
            "from __future__ import annotations\n"
            "\n"
            "import torch\n"
            "from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig\n"
            "\n"
            "\n"
            "class _Bench(BaseBenchmark):\n"
            "    allow_cpu = True\n"
            "\n"
            "    def __init__(self) -> None:\n"
            "        super().__init__()\n"
            "        self.device = torch.device('cpu')\n"
            "        self.output = None\n"
            "\n"
            "    def setup(self) -> None:\n"
            "        torch.backends.cudnn.deterministic = True\n"
            "        self.output = torch.zeros(1, device=self.device)\n"
            "\n"
            "    def benchmark_fn(self) -> None:\n"
            "        if self.output is None:\n"
            "            raise RuntimeError('missing output')\n"
            "        self.output = self.output + 1\n"
            "\n"
            "    def get_verify_output(self):\n"
            "        if self.output is None:\n"
            "            raise RuntimeError('missing output')\n"
            "        return self.output\n"
            "\n"
            "    def get_input_signature(self):\n"
            "        return {\n"
            "            'shapes': {'output': (1,)},\n"
            "            'dtypes': {'output': 'torch.float32'},\n"
            "            'batch_size': 1,\n"
            "            'parameter_count': 0,\n"
            "        }\n"
            "\n"
            "    def get_verify_inputs(self):\n"
            "        return {'dummy': torch.zeros(1)}\n"
            "\n"
            "    def get_output_tolerance(self):\n"
            "        return (0.0, 0.0)\n"
            "\n"
            "    def validate_result(self):\n"
            "        return None\n"
            "\n"
            "    def get_config(self) -> BenchmarkConfig:\n"
            "        return BenchmarkConfig(iterations=1, warmup=0, device=self.device)\n"
            "\n"
            "\n"
            "def get_benchmark():\n"
            "    return _Bench()\n"
        ),
    )

    results = audit_directory(tmp_path)
    result = _get_single_result(results)

    assert result["status"] == "needs_work"
    compliance = result["compliance"]
    assert compliance["determinism_toggles_present"] is True
    assert compliance["no_determinism_enable_without_justification"] is False
    warnings = result.get("warnings") or []
    assert any("Determinism enabled without" in msg for msg in warnings)


def test_audit_allows_determinism_when_explicitly_justified(tmp_path: Path) -> None:
    _write_benchmark(
        tmp_path,
        "baseline_determinism_on_justified.py",
        (
            "# aisp: allow_determinism needed for test coverage\n"
            "from __future__ import annotations\n"
            "\n"
            "import torch\n"
            "from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig\n"
            "\n"
            "\n"
            "class _Bench(BaseBenchmark):\n"
            "    allow_cpu = True\n"
            "\n"
            "    def __init__(self) -> None:\n"
            "        super().__init__()\n"
            "        self.device = torch.device('cpu')\n"
            "        self.output = None\n"
            "\n"
            "    def setup(self) -> None:\n"
            "        torch.backends.cudnn.deterministic = True\n"
            "        self.output = torch.zeros(1, device=self.device)\n"
            "\n"
            "    def benchmark_fn(self) -> None:\n"
            "        if self.output is None:\n"
            "            raise RuntimeError('missing output')\n"
            "        self.output = self.output + 1\n"
            "\n"
            "    def get_verify_output(self):\n"
            "        if self.output is None:\n"
            "            raise RuntimeError('missing output')\n"
            "        return self.output\n"
            "\n"
            "    def get_input_signature(self):\n"
            "        return {\n"
            "            'shapes': {'output': (1,)},\n"
            "            'dtypes': {'output': 'torch.float32'},\n"
            "            'batch_size': 1,\n"
            "            'parameter_count': 0,\n"
            "        }\n"
            "\n"
            "    def get_verify_inputs(self):\n"
            "        return {'dummy': torch.zeros(1)}\n"
            "\n"
            "    def get_output_tolerance(self):\n"
            "        return (0.0, 0.0)\n"
            "\n"
            "    def validate_result(self):\n"
            "        return None\n"
            "\n"
            "    def get_config(self) -> BenchmarkConfig:\n"
            "        return BenchmarkConfig(iterations=1, warmup=0, device=self.device)\n"
            "\n"
            "\n"
            "def get_benchmark():\n"
            "    return _Bench()\n"
        ),
    )

    results = audit_directory(tmp_path)
    result = _get_single_result(results)

    assert result["status"] == "compliant"
    compliance = result["compliance"]
    assert compliance["determinism_toggles_present"] is True
    assert compliance["no_determinism_enable_without_justification"] is True
    warnings = result.get("warnings") or []
    assert any("Determinism toggles detected" in msg for msg in warnings)
    assert not any("Determinism enabled without" in msg for msg in warnings)


def test_audit_does_not_require_justification_for_disabling_determinism(tmp_path: Path) -> None:
    _write_benchmark(
        tmp_path,
        "baseline_determinism_off.py",
        (
            "from __future__ import annotations\n"
            "\n"
            "import torch\n"
            "from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig\n"
            "\n"
            "\n"
            "class _Bench(BaseBenchmark):\n"
            "    allow_cpu = True\n"
            "\n"
            "    def __init__(self) -> None:\n"
            "        super().__init__()\n"
            "        self.device = torch.device('cpu')\n"
            "        self.output = None\n"
            "\n"
            "    def setup(self) -> None:\n"
            "        torch.backends.cudnn.deterministic = False\n"
            "        torch.use_deterministic_algorithms(False)\n"
            "        self.output = torch.zeros(1, device=self.device)\n"
            "\n"
            "    def benchmark_fn(self) -> None:\n"
            "        if self.output is None:\n"
            "            raise RuntimeError('missing output')\n"
            "        self.output = self.output + 1\n"
            "\n"
            "    def get_verify_output(self):\n"
            "        if self.output is None:\n"
            "            raise RuntimeError('missing output')\n"
            "        return self.output\n"
            "\n"
            "    def get_input_signature(self):\n"
            "        return {\n"
            "            'shapes': {'output': (1,)},\n"
            "            'dtypes': {'output': 'torch.float32'},\n"
            "            'batch_size': 1,\n"
            "            'parameter_count': 0,\n"
            "        }\n"
            "\n"
            "    def get_verify_inputs(self):\n"
            "        return {'dummy': torch.zeros(1)}\n"
            "\n"
            "    def get_output_tolerance(self):\n"
            "        return (0.0, 0.0)\n"
            "\n"
            "    def validate_result(self):\n"
            "        return None\n"
            "\n"
            "    def get_config(self) -> BenchmarkConfig:\n"
            "        return BenchmarkConfig(iterations=1, warmup=0, device=self.device)\n"
            "\n"
            "\n"
            "def get_benchmark():\n"
            "    return _Bench()\n"
        ),
    )

    results = audit_directory(tmp_path)
    result = _get_single_result(results)

    assert result["status"] == "compliant"
    compliance = result["compliance"]
    assert compliance["determinism_toggles_present"] is True
    assert compliance["no_determinism_enable_without_justification"] is True
    warnings = result.get("warnings") or []
    assert any("Determinism toggles detected" in msg for msg in warnings)
    assert not any("Determinism enabled without" in msg for msg in warnings)

