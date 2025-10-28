# Power Efficiency Baselines
## Tokens per Joule Metrics for NVIDIA B200 Workloads

**Last Updated**: October 28, 2025  
**Hardware**: 8x NVIDIA B200 GPUs (180GB HBM3e each)  
**Purpose**: Establish reproducible power efficiency baselines for production planning

---

## Executive Summary

This document provides validated power efficiency metrics (tokens/J) for common AI workloads on NVIDIA B200 hardware. Use these baselines to:

- **Estimate operational costs** for production deployments
- **Compare configurations** (FP16 vs FP8, batch sizes, model sizes)
- **Set efficiency targets** for optimization efforts
- **Justify hardware** investments with ROI analysis

### Key Findings

| Workload | Precision | Tokens/Joule | Cost/1M Tokens | Notes |
|----------|-----------|--------------|----------------|--------|
| GPT-8B Inference | FP16 | **TBD** | **TBD** | Baseline configuration |
| GPT-8B Inference | FP8 | **TBD** | **TBD** | 1.5-2x efficiency gain |
| GPT-40B Inference (8-GPU TP) | FP16 | **TBD** | **TBD** | Large model baseline |
| GPT-40B Inference (8-GPU TP) | FP8 | **TBD** | **TBD** | Best efficiency for large models |
| MoE-16x8B | FP16 | **TBD** | **TBD** | Sparse activation benefits |

**Note**: Values marked **TBD** will be populated after running `./run_comprehensive_8gpu_benchmark.sh`

---

## Measurement Methodology

### Tools Used

1. **Power Monitoring**: `tools/power_monitor.py`
   - Samples GPU power via NVML at 100ms intervals
   - Captures per-GPU and aggregate power
   - Measures total energy (Joules) for workload duration

2. **Throughput Measurement**: Inference/training scripts
   - Reports tokens/second throughput
   - Accounts for warmup period (excluded from power measurement)

3. **Cost Calculation**: `tools/calculate_cost_per_token.py`
   - Converts energy to cost using electricity rates
   - Compares to cloud API pricing
   - Calculates break-even point

### Measurement Protocol

```bash
# 1. Run workload with power monitoring
python tools/power_monitor.py \
    --interval 0.1 \
    --output power_metrics.json \
    --command "python ch16/test_gpt_large_optimized.py \
        --model-size 8B \
        --batch-size 16 \
        --seq-len 2048 \
        --iterations 100 \
        --warmup 10 \
        --output throughput.json"

# 2. Calculate efficiency metrics
python tools/calculate_cost_per_token.py \
    --power-json power_metrics.json \
    --throughput-file throughput.json \
    --output cost_analysis.md
```

### Validation Criteria

**✅ Valid Measurement**:
- Minimum duration: 60 seconds (excludes warmup)
- Minimum iterations: 50
- GPU utilization: >70% average
- Power variance: <5% coefficient of variation
- No thermal throttling detected

**❌ Invalid Measurement** (repeat test):
- Duration <60s
- GPU utilization <70%
- Power readings inconsistent
- Thermal throttling occurred

---

## Baseline Results

### Single-GPU Inference (B200)

#### GPT-8B Model

| Configuration | Batch | Seq Len | Throughput (tok/s) | Avg Power (W) | Tokens/Joule | Cost/1M Tokens |
|---------------|-------|---------|-------------------|---------------|--------------|----------------|
| FP16 Baseline | 8 | 2048 | **TBD** | **TBD** | **TBD** | **TBD** |
| FP16 Optimized | 16 | 2048 | **TBD** | **TBD** | **TBD** | **TBD** |
| FP8 (TE) | 16 | 2048 | **TBD** | **TBD** | **TBD** | **TBD** |
| FP8 + Compile | 16 | 2048 | **TBD** | **TBD** | **TBD** | **TBD** |

**Key Insights** (to be filled):
- FP8 provides **X%** efficiency improvement over FP16
- Larger batch sizes improve efficiency up to batch=**Y**
- torch.compile adds **Z%** efficiency gain

#### GPT-40B Model (Single GPU, Limited Context)

| Configuration | Batch | Seq Len | Throughput (tok/s) | Avg Power (W) | Tokens/Joule | Cost/1M Tokens |
|---------------|-------|---------|-------------------|---------------|--------------|----------------|
| FP16 | 2 | 1024 | **TBD** | **TBD** | **TBD** | **TBD** |
| FP8 | 4 | 1024 | **TBD** | **TBD** | **TBD** | **TBD** |

**Note**: Single-GPU 40B limited by memory; 8-GPU tensor parallel recommended

---

### Multi-GPU Inference (8x B200)

#### GPT-40B Model (8-GPU Tensor Parallel)

| Configuration | Batch | Seq Len | Throughput (tok/s) | Avg Power (W) | Tokens/Joule | Cost/1M Tokens | Efficiency vs 1-GPU |
|---------------|-------|---------|-------------------|---------------|--------------|----------------|---------------------|
| FP16 TP8 | 8 | 4096 | **TBD** | **TBD** | **TBD** | **TBD** | **TBD**x |
| FP16 TP8 | 16 | 4096 | **TBD** | **TBD** | **TBD** | **TBD** | **TBD**x |
| FP8 TP8 | 16 | 4096 | **TBD** | **TBD** | **TBD** | **TBD** | **TBD**x |
| FP8 TP8 | 32 | 4096 | **TBD** | **TBD** | **TBD** | **TBD** | **TBD**x |

**Key Insights** (to be filled):
- 8-GPU tensor parallel achieves **X%** scaling efficiency
- NVLink communication overhead: **Y%** of total power
- Optimal batch size for efficiency: **Z**

#### MoE-16x8B Model (8-GPU Expert Sharding)

| Configuration | Batch | Seq Len | Experts | Throughput (tok/s) | Avg Power (W) | Tokens/Joule | Cost/1M Tokens |
|---------------|-------|---------|---------|-------------------|---------------|--------------|----------------|
| FP16 | 16 | 2048 | 16 | **TBD** | **TBD** | **TBD** | **TBD** |
| FP16 | 32 | 2048 | 16 | **TBD** | **TBD** | **TBD** | **TBD** |
| FP8 | 32 | 2048 | 16 | **TBD** | **TBD** | **TBD** | **TBD** |

**Key Insights** (to be filled):
- MoE sparse activation saves **X%** power vs dense 130B model
- Expert routing overhead: **Y%** of total power
- Optimal expert placement: **Z**

---

### Long-Context Inference (16K tokens)

| Model | Precision | Batch | Seq Len | Throughput (tok/s) | Avg Power (W) | Tokens/Joule | Cost/1M Tokens |
|-------|-----------|-------|---------|-------------------|---------------|--------------|----------------|
| GPT-8B | FP16 | 4 | 16384 | **TBD** | **TBD** | **TBD** | **TBD** |
| GPT-8B | FP8 | 8 | 16384 | **TBD** | **TBD** | **TBD** | **TBD** |
| GPT-40B (TP8) | FP16 | 2 | 16384 | **TBD** | **TBD** | **TBD** | **TBD** |
| GPT-40B (TP8) | FP8 | 4 | 16384 | **TBD** | **TBD** | **TBD** | **TBD** |

**Key Insights** (to be filled):
- FlexAttention reduces power by **X%** for long contexts
- Memory bandwidth becomes bottleneck at **Y** tokens
- FP8 efficiency gain increases to **Z%** for long sequences

---

## Cost Analysis

### Electricity Cost Scenarios

Baseline assumptions:
- **GPU idle power**: ~50W per B200
- **System overhead**: CPU, memory, cooling = 20% of GPU power
- **Power supply efficiency**: 95%

#### Scenario 1: US Data Center ($0.10/kWh)

| Workload | Throughput (tok/s) | Power (W) | Cost/1M Tokens | Cost/Hour |
|----------|-------------------|-----------|----------------|-----------|
| GPT-8B FP16 | **TBD** | **TBD** | **TBD** | **TBD** |
| GPT-8B FP8 | **TBD** | **TBD** | **TBD** | **TBD** |
| GPT-40B TP8 FP16 | **TBD** | **TBD** | **TBD** | **TBD** |
| GPT-40B TP8 FP8 | **TBD** | **TBD** | **TBD** | **TBD** |

#### Scenario 2: EU Data Center ($0.20/kWh)

| Workload | Throughput (tok/s) | Power (W) | Cost/1M Tokens | Cost/Hour |
|----------|-------------------|-----------|----------------|-----------|
| GPT-8B FP16 | **TBD** | **TBD** | **TBD** | **TBD** |
| GPT-8B FP8 | **TBD** | **TBD** | **TBD** | **TBD** |
| GPT-40B TP8 FP16 | **TBD** | **TBD** | **TBD** | **TBD** |
| GPT-40B TP8 FP8 | **TBD** | **TBD** | **TBD** | **TBD** |

#### Scenario 3: High-Cost Region ($0.30/kWh)

| Workload | Throughput (tok/s) | Power (W) | Cost/1M Tokens | Cost/Hour |
|----------|-------------------|-----------|----------------|-----------|
| GPT-8B FP16 | **TBD** | **TBD** | **TBD** | **TBD** |
| GPT-8B FP8 | **TBD** | **TBD** | **TBD** | **TBD** |
| GPT-40B TP8 FP16 | **TBD** | **TBD** | **TBD** | **TBD** |
| GPT-40B TP8 FP8 | **TBD** | **TBD** | **TBD** | **TBD** |

### Break-Even Analysis vs Cloud APIs

#### GPT-4 Class Model (Comparable to GPT-40B)

| Provider | API Cost/1M Tokens | On-Prem Cost/1M Tokens | Tokens for Break-Even | Time to Break-Even |
|----------|-------------------|----------------------|---------------------|-------------------|
| OpenAI GPT-4 Turbo | $10.00 | **TBD** | **TBD** billion | **TBD** months |
| Anthropic Claude 3 | $15.00 | **TBD** | **TBD** billion | **TBD** months |
| Google Gemini Pro | $7.00 | **TBD** | **TBD** billion | **TBD** months |

**Assumptions**:
- 8x B200 hardware cost: $200,000 (estimated)
- 3-year depreciation
- Include: electricity, cooling, maintenance, rack space
- Exclude: network, storage, labor

#### GPT-3.5 Class Model (Comparable to GPT-8B)

| Provider | API Cost/1M Tokens | On-Prem Cost/1M Tokens | Tokens for Break-Even | Time to Break-Even |
|----------|-------------------|----------------------|---------------------|-------------------|
| OpenAI GPT-3.5 Turbo | $0.50 | **TBD** | **TBD** billion | **TBD** months |
| Open-source APIs | $0.20-0.70 | **TBD** | **TBD** billion | **TBD** months |

---

## Optimization Recommendations

### Configuration Guidelines

Based on collected metrics, recommended configurations for different priorities:

#### Priority: Maximum Efficiency (lowest cost/token)

```bash
# GPT-8B
python ch16/test_gpt_large_optimized.py \
    --model-size 8B \
    --batch-size 32 \
    --seq-len 2048 \
    --fp8-mode transformer-engine \
    --compile-mode default \
    --attention-backend flex

# Expected: **TBD** tokens/J, **TBD** $/1M tokens
```

#### Priority: Balanced (efficiency + latency)

```bash
# GPT-40B on 8-GPU
torchrun --nproc_per_node=8 ch16/test_gpt_large_optimized.py \
    --model-size 40B \
    --batch-size 16 \
    --seq-len 4096 \
    --fp8-mode transformer-engine \
    --skip-torch-compile \
    --attention-backend flex

# Expected: **TBD** tokens/J, **TBD** ms latency
```

#### Priority: Maximum Throughput

```bash
# GPT-8B with aggressive batching
python ch16/test_gpt_large_optimized.py \
    --model-size 8B \
    --batch-size 64 \
    --seq-len 2048 \
    --fp8-mode transformer-engine \
    --compile-mode max-autotune

# Expected: **TBD** tokens/s, **TBD** tokens/J
```

### Power Optimization Techniques

#### 1. Dynamic Voltage/Frequency Scaling

```bash
# Reduce GPU clock for power savings (trading performance)
nvidia-smi -lgc 1200  # Lock GPU clock to 1200 MHz (vs 1980 MHz default)

# Expected: 20-30% power reduction, 15-20% performance loss
# Net efficiency gain: 5-10% tokens/J
```

#### 2. Batch Size Tuning

**Finding the sweet spot**:

```python
# Sweep batch sizes to find optimal efficiency
for batch_size in [4, 8, 16, 32, 64]:
    run_benchmark(batch_size=batch_size)
    # Record: throughput, power, tokens/J

# Typically optimal at: **TBD** (to be determined from benchmarks)
```

#### 3. Precision Selection

**Decision Matrix**:

| Metric | FP16 | FP8 (TE) | Recommendation |
|--------|------|----------|----------------|
| Throughput | 1.0x | 1.5-2.0x | FP8 for throughput |
| Power | Baseline | -10% to -15% | FP8 for efficiency |
| Accuracy | Baseline | -0.5% to -1% perplexity | FP16 if accuracy critical |
| Memory | Baseline | -40% | FP8 for large models |
| **Tokens/J** | 1.0x | **1.6-2.2x** | **FP8 wins** |

---

## Continuous Monitoring

### Recommended Metrics

Track these metrics in production:

1. **Tokens/Joule**: Primary efficiency metric
2. **Cost/1M Tokens**: Business metric
3. **Power/GPU**: Capacity planning
4. **Throughput**: Performance metric
5. **Idle Power**: Waste detection

### Regression Detection

Alert if efficiency degrades:

```yaml
alert: PowerEfficiencyRegression
expr: |
  (tokens_per_joule{workload="gpt8b"} / 
   tokens_per_joule{workload="gpt8b"} offset 7d) < 0.9
for: 1h
severity: warning
annotations:
  summary: "Power efficiency dropped 10%+ vs last week"
  action: "Check for config changes, thermal issues, or hardware degradation"
```

### Monthly Reporting

Generate monthly efficiency reports:

```bash
# tools/generate_efficiency_report.py
python tools/generate_efficiency_report.py \
    --start-date 2025-10-01 \
    --end-date 2025-10-31 \
    --output october_2025_efficiency.pdf

# Includes:
# - Tokens/J trends over time
# - Cost/1M tokens breakdown
# - Comparison to baselines
# - Optimization opportunities
```

---

## Data Collection Plan

### Phase 1: Initial Baselines (Week 1)

**Tasks**:
1. ✅ Set up power monitoring infrastructure
2. ⏳ Run benchmark suite: `./run_comprehensive_8gpu_benchmark.sh`
3. ⏳ Collect results and populate TBD values in this document
4. ⏳ Validate measurements meet quality criteria

**Deliverables**:
- Completed baseline table (all TBD values filled)
- Raw data files in `benchmark_results/`
- Validation report

### Phase 2: Extended Workloads (Week 2-3)

**Tasks**:
1. Test vision models (ViT, CLIP)
2. Test diffusion models (Stable Diffusion variants)
3. Test recommender systems
4. Test training workloads

**Deliverables**:
- Extended baseline tables for new workloads
- Architecture-specific efficiency guides

### Phase 3: Optimization Sweep (Week 4)

**Tasks**:
1. Tune batch sizes for each workload
2. Test DVFS settings
3. Evaluate mixed-precision strategies
4. Document best practices

**Deliverables**:
- Optimization playbook
- Configuration templates
- Cost calculator tool

---

## Running the Benchmark Suite

To populate this document with actual measurements:

```bash
# 1. Run comprehensive benchmark suite
./run_comprehensive_8gpu_benchmark.sh

# This will generate: 8gpu_benchmark_results_YYYYMMDD_HHMMSS/
# Including:
#   - power_metrics_*.json (power consumption)
#   - *_results.json (throughput data)
#   - cost_analysis.md (cost calculations)

# 2. Extract metrics and update this document
python tools/update_power_baselines.py \
    --results-dir 8gpu_benchmark_results_YYYYMMDD_HHMMSS/ \
    --output docs/power_efficiency_baselines.md

# 3. Commit updated baselines
git add docs/power_efficiency_baselines.md
git commit -m "Update power efficiency baselines with $(date +%Y-%m-%d) measurements"
```

---

## Frequently Asked Questions

### Q: Why are tokens/J values TBD?

**A**: These require actual hardware measurements on production 8x B200 systems. The infrastructure is ready; we're awaiting completion of `run_comprehensive_8gpu_benchmark.sh` to populate real numbers.

### Q: How do I compare different model sizes?

**A**: Use "tokens/$" instead of "tokens/J" to account for hardware cost amortization. Larger models need more GPUs but may have better tokens/$ for high-volume scenarios.

### Q: What about training efficiency?

**A**: This document focuses on inference. Training efficiency (samples/J) will be covered in a separate document: `docs/training_efficiency_baselines.md`.

### Q: How often should baselines be updated?

**A**: 
- **Quarterly**: For production systems
- **After each major update**: PyTorch, CUDA, drivers, model changes
- **Ad-hoc**: When investigating performance regressions

---

## Related Documentation

- [8x B200 Load Testing Guide](8xb200_load_testing_guide.md)
- [Cost Calculator Tool](../tools/calculate_cost_per_token.py)
- [Power Monitoring Tool](../tools/power_monitor.py)
- [Architecture Guides](architecture_guides.md)
- [MoE Deployment Playbook](moe_deployment_playbook.md)

---

## Changelog

### 2025-10-28
- Initial document created with measurement methodology
- TBD placeholders for actual measurements
- Infrastructure ready for data collection

### Future Updates
- ⏳ Populate baseline tables with measured values
- ⏳ Add vision and diffusion model results
- ⏳ Include training efficiency metrics
- ⏳ Expand cost analysis scenarios

---

**Status**: 🔶 **Awaiting Measurements**

Run `./run_comprehensive_8gpu_benchmark.sh` to complete this document.

**Estimated Time to Complete**: 2-3 hours for full benchmark suite

