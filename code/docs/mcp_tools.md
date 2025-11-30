# MCP Tools Reference

Complete reference for the 76 MCP tools available in the `aisp` MCP server.

## Quick Start

```bash
# Start the MCP server
python -m mcp.mcp_server --serve

# Or configure via mcp.json for IDE integration
```

Tool descriptions returned by `tools/list` (or `python -m mcp.mcp_server --list`) now embed **Inputs**, **Outputs**, and **Expectations** so MCP clients can surface guidance on parameters, the JSON envelope that comes back, and runtime/side-effect hints.

**Response Format:** All tools return a `text` content entry containing a JSON envelope.

---

## Benchmarking (17 tools)

Tools for running benchmarks, hardware tests, and performance verification.

| Tool | Description | Parameters |
|------|-------------|------------|
| `run_benchmarks` | Run benchmarks via the bench CLI with optional profiling/LLM analysis | `targets`, `profile`, `llm_analysis`, `apply_patches` |
| `verify_benchmarks` | Verify benchmarks via the bench CLI (smoke tests) | `targets` |
| `available_benchmarks` | List available benchmarks | - |
| `benchmark_targets` | List benchmark targets supported by the harness | - |
| `list_chapters` | List all discoverable chapters and labs | - |
| `benchmark_report` | Generate PDF/HTML report from benchmark results | - |
| `benchmark_export` | Export benchmark results to csv/markdown/json | - |
| `benchmark_compare_runs` | Diff two benchmark JSON files and show speedup deltas | `baseline`, `optimized` |

### Hardware Micro-Benchmarks

| Tool | Description | Parameters |
|------|-------------|------------|
| `test_speed` | Quick performance sanity checks | - |
| `test_disk` | Disk I/O benchmark (sequential) | `file_size_mb`, `block_size_kb`, `tmp_dir` |
| `test_pcie` | PCIe H2D/D2H bandwidth benchmark | `size_mb`, `iters` |
| `test_mem_hierarchy` | Memory hierarchy stride test on GPU | `size_mb`, `stride` |
| `test_tensor_core` | Tensor Core matmul throughput test | `size`, `precision` |
| `test_sfu` | Special function unit benchmark (sin/cos) | `elements` |
| `test_network` | Network throughput tests | - |
| `test_network_loopback` | Loopback TCP throughput test | `size_mb`, `port` |
| `test_roofline` | Stride sweep ASCII roofline for memory | - |

---

## GPU (5 tools)

Tools for GPU information, topology, and hardware details.

| Tool | Description | Parameters |
|------|-------------|------------|
| `gpu_info` | Get detailed GPU information (name, memory, temperature, power) | - |
| `gpu_bandwidth` | Run GPU memory bandwidth test (actual vs theoretical) | - |
| `gpu_topology` | Get multi-GPU topology showing NVLink/PCIe connections | - |
| `gpu_topology_matrix` | Get GPU/NUMA topology matrix (nvidia-smi topo -m) | - |
| `gpu_power` | Get current GPU power consumption and limits | - |

---

## System (13 tools)

Tools for system information, dependencies, and environment analysis.

| Tool | Description | Parameters |
|------|-------------|------------|
| `system_software` | Get software stack info (PyTorch, CUDA, Python versions) | - |
| `system_dependencies` | Check health of installed ML/AI dependencies | - |
| `system_context` | Get full system context for AI analysis | - |
| `system_capabilities` | Get hardware capabilities summary | - |
| `system_parameters` | Inspect kernel/system parameters (swappiness, dirty ratios) | - |
| `container_limits` | Inspect container/cgroup limits | - |
| `cpu_memory_analysis` | Analyze CPU/memory hierarchy (NUMA, caches, TLB, hugepages) | - |
| `full_system_analysis` | Complete system analysis with recommendations | - |
| `status` | Quick system status: GPU, software, AI backend | - |
| `triage` | Quick triage snapshot: status + context summary | - |
| `context_summary` | Lightweight system context summary | - |
| `context_full` | Full detailed system context | - |
| `ai_status` | Check AI/LLM backend availability | - |

---

## Profiling (10 tools)

Tools for performance profiling with Nsight Systems and Nsight Compute.

| Tool | Description | Parameters |
|------|-------------|------------|
| `profile_nsys` | Run Nsight Systems on a command | `command`, `output_dir`, `output_name`, `preset`, `trace_cuda`, `trace_nvtx`, `trace_osrt`, `trace_forks`, `full_timeline` |
| `profile_ncu` | Run Nsight Compute on a command | `command`, `output_dir`, `output_name`, `workload_type`, `kernel_filter` |
| `nsys_summary` | Summarize an existing .nsys-rep or CSV | `report_path` |
| `compare_nsys` | Compare baseline vs optimized Nsight Systems reports | `profiles_dir` |
| `compare_ncu` | Compare baseline vs optimized Nsight Compute reports | `profiles_dir` |
| `nsys_ncu_available` | Check availability of Nsight tools | - |
| `profile_flame` | Get flame graph data for execution time breakdown | - |
| `profile_memory` | Get memory allocation timeline | - |
| `profile_kernels` | Get kernel execution breakdown (CUDA kernel times) | - |
| `profile_roofline` | Get roofline model data (compute vs memory bound) | - |

---

## Analysis (5 tools)

Tools for performance analysis and bottleneck detection.

| Tool | Description | Parameters |
|------|-------------|------------|
| `analyze_bottlenecks` | Identify performance bottlenecks in the current workload | `analysis_type`, `mode` |
| `analyze_pareto` | Find Pareto-optimal benchmarks (throughput/latency/memory) | - |
| `analyze_scaling` | Analyze how optimizations scale with workload size | - |
| `analyze_stacking` | Show which optimization techniques work well together | - |
| `analyze_whatif` | What-if analysis: find optimizations meeting constraints | `max_latency_ms`, `max_vram_gb`, `min_throughput` |

---

## Optimization (3 tools)

Tools for optimization recommendations and technique selection.

| Tool | Description | Parameters |
|------|-------------|------------|
| `recommend` | Get optimization recommendations for a model configuration | `model_size`, `gpus`, `goal` |
| `optimize_roi` | Calculate ROI of optimization techniques | - |
| `optimize_techniques` | Get list of all available optimization techniques | - |

---

## Distributed (2 tools)

Tools for distributed training configuration.

| Tool | Description | Parameters |
|------|-------------|------------|
| `distributed_plan` | Plan parallelism strategy (DP/TP/PP/FSDP) | `model_size`, `gpus`, `nodes` |
| `distributed_nccl` | Get NCCL tuning recommendations | `nodes`, `gpus` |

---

## Inference (2 tools)

Tools for inference optimization.

| Tool | Description | Parameters |
|------|-------------|------------|
| `inference_vllm` | Generate vLLM configuration for inference | `model`, `target` |
| `inference_quantization` | Get quantization recommendations (FP8, INT8, INT4) | `model_size` |

---

## AI/LLM (2 tools)

Tools for AI-powered performance advice.

| Tool | Description | Parameters |
|------|-------------|------------|
| `ask` | Ask a performance optimization question (with book citations) | `question` |
| `explain` | Explain a GPU/AI performance concept | `concept` |

---

## HuggingFace (2 tools)

Tools for HuggingFace model discovery.

| Tool | Description | Parameters |
|------|-------------|------------|
| `hf_search` | Search HuggingFace for models | `query`, `limit` |
| `hf_trending` | Get trending models on HuggingFace | `task` |

---

## Cluster & Cost (2 tools)

Tools for cluster job generation and cost estimation.

| Tool | Description | Parameters |
|------|-------------|------------|
| `cluster_slurm` | Generate SLURM script for cluster job submission | `model`, `nodes`, `gpus` |
| `cost_estimate` | Estimate cloud costs for training/inference | `model_size`, `training_tokens`, `provider` |

---

## Code Analysis (7 tools)

Tools for analyzing code patterns and GPU utilization.

| Tool | Description | Parameters |
|------|-------------|------------|
| `warp_divergence` | Analyze code for warp divergence patterns | `code` |
| `bank_conflicts` | Analyze shared memory bank conflicts | `stride`, `element_size` |
| `memory_access` | Analyze memory access patterns for coalescing | `stride`, `element_size` |
| `comm_overlap` | Analyze communication overlap for a model | `model` |
| `data_loading` | Analyze data loading pipeline | - |
| `energy_analysis` | Analyze energy efficiency | - |
| `predict_scaling` | Predict scaling behavior for model size and GPU count | `model_size`, `gpus` |

---

## Export (3 tools)

Tools for exporting benchmark results.

| Tool | Description | Parameters |
|------|-------------|------------|
| `export_csv` | Export benchmarks to CSV | `detailed` |
| `export_pdf` | Export benchmarks to PDF report | - |
| `export_html` | Export benchmarks to HTML report | - |

---

## Utility (3 tools)

Helper tools for navigation and planning.

| Tool | Description | Parameters |
|------|-------------|------------|
| `help` | Get help and tool suggestions | `query` |
| `suggest_tools` | Get ranked tool suggestions based on intent | `query` |
| `launch_plan` | Generate a torchrun launch plan (TP/PP/DP layout) | - |

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AISP_MCP_PREVIEW_LIMIT` | Max characters in response previews | 10000 |
| `AISP_MCP_PREVIEW_ITEMS` | Max items in list/dict previews | 50 |

### Response Format

All tools return JSON responses with the following structure:

```json
{
  "status": "success" | "error",
  "data": { ... },
  "tool": "tool_name",
  "timestamp": "ISO timestamp",
  "next_steps": ["suggested", "follow-up", "tools"]
}
```

The `isError` field in the MCP response mirrors the `status` field.

---

## Examples

### Run benchmarks with profiling
```
Tool: run_benchmarks
Parameters: { "targets": ["ch7"], "profile": "minimal" }
```

### Get optimization recommendations for 70B model
```
Tool: recommend
Parameters: { "model_size": 70, "gpus": 8, "goal": "throughput" }
```

### Profile a training script with Nsight Systems
```
Tool: profile_nsys
Parameters: { "command": ["python", "train.py", "--batch", "32"] }
```

### What-if analysis with constraints
```
Tool: analyze_whatif
Parameters: { "max_vram_gb": 24, "max_latency_ms": 50 }
```

### Ask a performance question
```
Tool: ask
Parameters: { "question": "Why is my attention kernel slow on H100?" }
```
