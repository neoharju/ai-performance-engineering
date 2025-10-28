# Ready to Run: 8x B200 Hardware Validation Guide

**Status**: 🟢 **All Infrastructure Complete**  
**Date**: October 28, 2025  
**Next Action**: Execute benchmarks on production hardware

---

## What Was Fixed

### 1. Load Test Import Error ✅

**Problem**: `ModuleNotFoundError: No module named 'ch16'`

**Solution**: 
- Updated `tools/orchestrate_8xb200_load_test.sh` to set PYTHONPATH
- Fixed all torchrun commands to use absolute paths
- Updated `run_comprehensive_8gpu_benchmark.sh` with proper path handling

**Validation**:
```bash
# Test the fix
./tools/orchestrate_8xb200_load_test.sh 30 50 test_run
```

---

## New Tools Created

### 1. NVLink Bandwidth Capture ✅

**File**: `tools/capture_nvlink_during_inference.sh`

**Purpose**: Capture NVLink bandwidth and latency metrics during inference stress tests

**Usage**:
```bash
# Capture NVLink metrics during 60s inference workload
./tools/capture_nvlink_during_inference.sh 60 nvlink_results inference

# Output: nvlink_results/
#   - nvlink_topology.txt
#   - nvlink_utilization.txt  
#   - nvlink_analysis.txt
#   - SUMMARY.md
```

### 2. Nsight Systems Profiling for 40B Models ✅

**File**: `tools/profile_40b_8gpu_nsight.sh`

**Purpose**: Profile large models on 8-GPU tensor parallel with Nsight Systems

**Usage**:
```bash
# Profile 40B model for 30 seconds
./tools/profile_40b_8gpu_nsight.sh 40B 30 profile_output

# Output: profile_output/
#   - profile_40B_8gpu.nsys-rep (open in Nsight UI)
#   - summary.txt
#   - gpu_kernel_summary.txt
#   - SUMMARY.md
```

**View Results**:
```bash
nsys-ui profile_output/profile_40B_8gpu.nsys-rep
```

---

## New Documentation Created

### 1. MoE Deployment Playbook ✅

**File**: `docs/moe_deployment_playbook.md`

**Contents**:
- Architecture considerations (replicated vs sharded vs hybrid)
- Deployment patterns (single-node, multi-node, disaggregated)
- **Routing telemetry**: Expert utilization, load balance, latency tracking
- **Autoscaling strategies**: Expert-level scaling, batch size adaptation, expert caching
- **Monitoring & alerting**: Prometheus metrics, Grafana dashboards, critical alerts
- Performance optimization techniques
- Cost optimization strategies
- Troubleshooting guide

**Key Sections**:
- How to track expert utilization with metrics
- Load balancing (Gini coefficient)
- Autoscaling based on expert load
- Kubernetes HPA configuration
- Production monitoring setup

### 2. Power Efficiency Baselines ✅

**File**: `docs/power_efficiency_baselines.md`

**Contents**:
- Measurement methodology (tools, protocol, validation criteria)
- Baseline result tables (currently with TBD placeholders)
- Cost analysis across electricity scenarios
- Break-even analysis vs cloud APIs
- Optimization recommendations
- Continuous monitoring setup

**To Be Populated**:
- Run `./run_comprehensive_8gpu_benchmark.sh` to fill TBD values
- Tables will show tokens/Joule for each workload
- Cost per 1M tokens at different electricity rates
- Break-even points vs OpenAI, Anthropic, Google

---

## Updated Scripts

### 1. Orchestration Script ✅

**File**: `tools/orchestrate_8xb200_load_test.sh`

**Changes**:
- Added PYTHONPATH export for imports
- Fixed absolute path to inference_server_load_test.py
- Improved error handling

### 2. Comprehensive Benchmark Suite ✅

**File**: `run_comprehensive_8gpu_benchmark.sh`

**Changes**:
- Set PYTHONPATH at startup
- All commands use absolute paths
- Optional tests gracefully skip if scripts missing
- Added NVLink capture test
- Enhanced power efficiency measurement
- Better error handling (continue on failures)
- Generates comprehensive summary

---

## How to Run Everything

### Step 1: Quick Load Test (5 minutes)

```bash
# Run a quick 2-minute load test to validate fixes
./tools/orchestrate_8xb200_load_test.sh 120 100 quick_test

# Check results
cat quick_test/SUMMARY.md
```

**Expected Output**:
- ✅ Load test completes successfully
- Power metrics captured
- Cost analysis generated
- System info collected

### Step 2: Full Benchmark Suite (2-3 hours)

```bash
# Run comprehensive benchmarks
./run_comprehensive_8gpu_benchmark.sh

# Results saved to: 8gpu_benchmark_results_YYYYMMDD_HHMMSS/
```

**What This Does**:
1. ✅ Multi-GPU tensor parallel validation
2. ✅ Inference server load test (2 minutes @ 100 QPS)
3. ✅ NVLink bandwidth benchmark
4. ⏳ Memory profiling (optional, if tools exist)
5. ⏳ Accuracy evaluation (optional)
6. ⏳ MoE benchmark (optional)
7. ⏳ 40B model on 8-GPU (optional)
8. ✅ Inference server stress test (60s)
9. ✅ Power efficiency analysis (GPT-8B)
10. ✅ NVLink bandwidth during stress

**Core Tests** (always run):
- Multi-GPU validation
- Load test with power monitoring
- Power efficiency analysis
- NVLink bandwidth capture

**Optional Tests** (skip gracefully if scripts missing):
- Memory profiling
- Accuracy evaluation
- MoE benchmarks
- 40B model tests

### Step 3: Deep Profiling (30 minutes)

```bash
# Profile 40B model with Nsight Systems
./tools/profile_40b_8gpu_nsight.sh 40B 30 nsight_profile

# Open in Nsight UI
nsys-ui nsight_profile/profile_40B_8gpu.nsys-rep
```

**Analyzes**:
- GPU utilization timeline
- NVLink communication patterns
- Kernel execution breakdown
- Memory transfer overhead
- NCCL operation latency

### Step 4: Capture NVLink Metrics (1 minute)

```bash
# Capture NVLink bandwidth during inference
./tools/capture_nvlink_during_inference.sh 60 nvlink_stress inference

# Review results
cat nvlink_stress/SUMMARY.md
cat nvlink_stress/nvlink_analysis.txt
```

---

## What to Do With Results

### 1. Update Power Efficiency Baselines

```bash
# After running benchmarks, extract metrics
RESULTS_DIR="8gpu_benchmark_results_YYYYMMDD_HHMMSS"

# View power consumption
cat ${RESULTS_DIR}/cost_analysis_8b.md

# Extract key metrics:
# - Throughput (tokens/sec)
# - Average power (W)
# - Tokens/Joule
# - Cost per 1M tokens

# Update docs/power_efficiency_baselines.md with actual values
# Replace all "TBD" placeholders
```

### 2. Update KNOWN_GAPS.md

Mark hardware validation as complete:

```markdown
### 6. Multi-GPU Support
**Status**: ✅ Validated on production 8x B200 hardware

**Progress**:
- ✅ Tensor-parallel correctness validated
- ✅ Load test completed successfully on 8x B200
- ✅ NVLink bandwidth measured under stress
- ✅ Power efficiency baselines established

**Results archived**: 8gpu_benchmark_results_YYYYMMDD_HHMMSS.tar.gz

**Outstanding**:
- ⏳ Extend to 32K token sequences (pending larger memory GPUs)
```

### 3. Archive Results

```bash
# Results are auto-archived to .tar.gz
RESULTS_DIR="8gpu_benchmark_results_YYYYMMDD_HHMMSS"

# Upload to shared storage (if available)
rsync -av ${RESULTS_DIR}.tar.gz storage:/shared/benchmarks/

# Or commit to git (if size permits)
git add ${RESULTS_DIR}
git commit -m "Add 8x B200 hardware validation results $(date +%Y-%m-%d)"
```

---

## Troubleshooting

### Issue: Load test still fails

**Check**:
```bash
# Verify PYTHONPATH is set
echo $PYTHONPATH  # Should include repo root

# Test import manually
python -c "from ch16.inference_serving_8xb200 import InferenceServer8GPU; print('OK')"

# If fails, set manually:
export PYTHONPATH=/home/ubuntu/dev/ai-performance-engineering/code:$PYTHONPATH
```

### Issue: Nsight profiling fails

**Solution**:
```bash
# Check if nsys is installed
which nsys

# If not found, install Nsight Systems:
# https://developer.nvidia.com/nsight-systems

# Or skip profiling and use basic benchmarks only
```

### Issue: Power monitoring not working

**Solution**:
```bash
# Install nvidia-ml-py3
pip install nvidia-ml-py3

# Test power monitoring
python tools/power_monitor.py -- sleep 5

# Check output for power readings
```

---

## Expected Timeline

| Task | Duration | Status |
|------|----------|--------|
| Quick load test | 5 min | ✅ Ready |
| Full benchmark suite | 2-3 hours | ✅ Ready |
| Nsight profiling | 30 min | ✅ Ready |
| NVLink capture | 1-5 min | ✅ Ready |
| Results analysis | 1 hour | Manual |
| Documentation update | 1 hour | Manual |

**Total**: ~4-5 hours for complete validation

---

## Success Criteria

### ✅ Load Test Passes

- All 8 GPUs participate
- No import errors
- Latency metrics collected (P50, P95, P99)
- Power consumption tracked
- Cost per token calculated

### ✅ Benchmark Suite Completes

- At least core tests succeed
- Results archived to .tar.gz
- Summary report generated
- No critical failures

### ✅ Documentation Updated

- `power_efficiency_baselines.md` filled with real values
- `KNOWN_GAPS.md` updated to reflect completion
- Results archived and accessible

---

## Quick Reference Commands

```bash
# 1. Quick validation (5 min)
./tools/orchestrate_8xb200_load_test.sh 120 100 quick_test

# 2. Full benchmarks (2-3 hours)
./run_comprehensive_8gpu_benchmark.sh

# 3. Profile 40B model (30 min)
./tools/profile_40b_8gpu_nsight.sh 40B 30 profile_results

# 4. NVLink stress test (1 min)
./tools/capture_nvlink_during_inference.sh 60 nvlink_results inference

# 5. Review results
cat quick_test/SUMMARY.md
cat 8gpu_benchmark_results_*/SUMMARY.md
cat nvlink_results/SUMMARY.md
```

---

## Additional Resources

- **Load Testing Guide**: `docs/8xb200_load_testing_guide.md`
- **MoE Deployment**: `docs/moe_deployment_playbook.md`
- **Power Efficiency**: `docs/power_efficiency_baselines.md`
- **Architecture Guides**: `docs/architecture_guides.md`
- **Known Gaps**: `KNOWN_GAPS.md`

---

## Next Steps After Validation

1. **Analyze Results**: Review all summaries and identify bottlenecks
2. **Optimize**: Apply recommendations from MoE playbook and power guide
3. **Document**: Update baselines and architecture guides
4. **Share**: Archive results and update team documentation
5. **Production**: Deploy validated configurations to production

---

**Status**: 🟢 Ready to Execute  
**Blockers**: None  
**Dependencies**: 8x B200 hardware available

**Execute**: `./run_comprehensive_8gpu_benchmark.sh`

