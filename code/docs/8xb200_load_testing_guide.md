# 8x B200 Load Testing Guide

This guide describes how to run comprehensive load tests on 8x NVIDIA B200 GPU systems.

## Prerequisites

### Hardware
- 8x NVIDIA B200 GPUs with NVLink/NVSwitch connectivity
- Sufficient system memory (512GB+ recommended)
- Fast storage for checkpoints and results

### Software
- CUDA 13.0+
- PyTorch 2.9+ with CUDA support
- Python 3.10+
- nvidia-ml-py3 (for power monitoring)

### Environment Setup

```bash
# Set optimal NCCL settings for B200
export NCCL_PROTO="Simple"
export NCCL_ALGO="Tree,Ring,NVLS"
export NCCL_NVLINK_C2C_ENABLE=1
export NCCL_NVLINK_TCE_ENABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_P2P_LEVEL="NVL"
export NCCL_NCHANNELS_PER_NET_PEER=8
```

## Quick Start

### Basic Load Test

Run a 5-minute load test at 100 QPS:

```bash
./tools/orchestrate_8xb200_load_test.sh 300 100
```

Arguments:
1. **Duration** (seconds): How long to run the test (default: 300)
2. **Target QPS**: Target queries per second (default: 100)
3. **Output directory**: Where to save results (default: auto-generated timestamp)

### Custom Configuration

```bash
# Run 10-minute test at 200 QPS
./tools/orchestrate_8xb200_load_test.sh 600 200 my_results

# Run stress test at 500 QPS
./tools/orchestrate_8xb200_load_test.sh 300 500 stress_test_results
```

## What Gets Tested

The orchestration script performs:

1. **Pre-flight Checks**
   - Validates 8 GPUs are available
   - Confirms B200 GPU model
   - Verifies NCCL configuration

2. **Power Monitoring**
   - Samples GPU power at 0.5s intervals
   - Tracks per-GPU and aggregate power
   - Calculates total energy consumption

3. **Load Test Execution**
   - Distributes load across 8 GPUs using torchrun
   - Simulates realistic inference traffic
   - Collects latency percentiles (p50, p95, p99)
   - Measures throughput (tokens/sec, QPS)

4. **System Metrics Collection**
   - GPU topology and NVLink status
   - CPU and memory configuration
   - Driver versions
   - Environment variables

5. **Cost Analysis**
   - Calculates cost per token
   - Estimates operating costs
   - Compares to API pricing

## Output Files

After the test completes, you'll find:

```
load_test_8xb200_YYYYMMDD_HHMMSS/
├── SUMMARY.md                    # Quick overview of results
├── load_test_results.json        # Detailed load test metrics
├── power_metrics.json            # Power consumption data
├── cost_analysis.md              # Detailed cost breakdown
├── cost_metrics.json             # Cost metrics (JSON)
├── system_info.txt               # Hardware configuration
├── load_test.log                 # Execution logs
├── power_monitor.log             # Power monitoring logs
└── cost_calculation.log          # Cost calculation logs
```

## Understanding Results

### Load Test Metrics

Key metrics in `load_test_results.json`:

- **actual_qps**: Achieved queries per second
- **tokens_per_sec**: Total token generation throughput
- **latency_p50_ms**: Median latency
- **latency_p95_ms**: 95th percentile latency
- **latency_p99_ms**: 99th percentile latency
- **total_requests**: Number of requests generated
- **completed_requests**: Number of requests completed

### Power Metrics

From `power_metrics.json`:

- **avg_watts**: Average power consumption across all GPUs
- **max_watts**: Peak power consumption
- **energy_joules**: Total energy consumed during test

### Cost Metrics

From `cost_analysis.md`:

- **Cost per token**: How much each token costs in electricity
- **Cost per 1M tokens**: Useful for comparing to API pricing
- **Tokens per dollar**: How many tokens you get per dollar
- **Operating cost**: Hourly cost to run the system

## Troubleshooting

### Test Fails Immediately

```bash
# Check GPU availability
nvidia-smi

# Verify distributed environment
torchrun --nproc_per_node=1 -m torch.distributed.run --help
```

### Low Throughput

- Check GPU utilization: `nvidia-smi dmon -s u`
- Verify NVLink is active: `nvidia-smi nvlink --status`
- Increase batch size or sequence length
- Profile with Nsight Systems: `./tools/profile_all_workloads.sh`

### High Latency

- Reduce target QPS
- Check for thermal throttling: `nvidia-smi -q -d TEMPERATURE`
- Monitor system resources: `htop`, `iotop`

### Power Monitoring Not Working

```bash
# Install pynvml
pip install nvidia-ml-py3

# Test power monitoring
python3 tools/power_monitor.py -- sleep 10
```

## Advanced Usage

### Running with Nsight Profiling

```bash
# Profile the first 30 seconds of a load test
nsys profile \
    --output=load_test_profile.nsys-rep \
    --duration=30 \
    --capture-range=cudaProfilerApi \
    torchrun --nproc_per_node=8 ch16/inference_server_load_test.py \
    --duration 300 --target-qps 100
```

### Custom Test Scenarios

Edit `ch16/inference_server_load_test.py` to customize:

- Prompt length distribution
- Generation length (`--max-new-tokens`)
- Temperature / sampling parameters
- Request arrival patterns

### Automated Regression Testing

Integrate into CI:

```bash
# Run load test and check for regressions
./tools/orchestrate_8xb200_load_test.sh 60 100 ci_results

# Compare to baseline
python3 tools/detect_regressions.py \
    --current ci_results/load_test_results.json \
    --baseline baseline/load_test_results.json \
    --fail-on-regression
```

## Performance Targets

Expected performance on 8x B200:

| Metric | Target | Notes |
|--------|--------|-------|
| Throughput | 10,000+ tokens/sec | Depends on model size and batch size |
| P95 Latency | <100ms | For online serving workloads |
| P99 Latency | <200ms | With 95%+ GPU utilization |
| Cost per 1M tokens | <$0.50 | Electricity only, at $0.16/kWh |

## Best Practices

1. **Warm-up Period**: Run for at least 1 minute to allow JIT compilation and caching
2. **Multiple Runs**: Run tests 3-5 times and average results
3. **Thermal Stability**: Ensure cooling is adequate before long runs
4. **Baseline Comparison**: Always compare to previous baselines
5. **Document Changes**: Track hardware, software, and configuration changes

## Archiving Results

For production validation:

```bash
# Archive results with git commit hash
COMMIT=$(git rev-parse --short HEAD)
./tools/orchestrate_8xb200_load_test.sh 300 100 "results_${COMMIT}"

# Upload to shared storage
rsync -av results_${COMMIT}/ storage:/shared/benchmarks/
```

## Next Steps

After successful load testing:

1. Review `SUMMARY.md` for high-level results
2. Check `cost_analysis.md` for operational cost projections
3. Compare to previous baselines using `detect_regressions.py`
4. Profile hotspots with `profile_all_workloads.sh`
5. Archive results for future reference

## Support

For issues or questions:
- Check logs in the output directory
- Review system_info.txt for configuration issues
- Profile with Nsight Systems for performance debugging
- Compare power metrics to thermal limits

## Related Documentation

- [Inference Serving Implementation](../ch16/inference_serving_8xb200.py)
- [Load Test Harness](../ch16/inference_server_load_test.py)
- [Power Monitoring](../tools/power_monitor.py)
- [Cost Analysis](../tools/calculate_cost_per_token.py)
- [Migration Guide](migration_to_b200.md)


