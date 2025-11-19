# Distributed Training Lab (DDP / FSDP / ZeRO)

- Targets (baseline/optimized pairs): `ddp`, `fsdp`, `zero1`, `zero2`, `zero3`.
- Entry points: `ddp.py`, `train_fsdp.py`, and `zero{1,2,3}.py` dispatch `--mode baseline|optimized`.
- Shared helpers live in `training_utils/` (torchrun harness, dataloaders, memory utils).
- Symmetric memory training: `symmem_training` (baseline disables symmetric memory, optimized enables it via `ch4/symmetric_memory_training_advanced.py`).

## Harness launch (torchrun)
Parse rank0 output only; world size is captured in results. All targets are marked multi-GPU so single-GPU runs are skipped instead of hanging.
```bash
PYTHONPATH=. python tools/cli/benchmark_cli.py run \
  --targets labs/train_distributed:ddp \
  --launch-via torchrun --nproc-per-node 2 \
  --iterations 50 \
  --target-extra-arg 'labs/train_distributed:ddp=--compile'
```
Add `--torchrun-env CUDA_VISIBLE_DEVICES=0,1` to pin devices or `--nnodes/--rdzv-endpoint` for multi-node rendezvous. Per-target overrides work via `--target-extra-arg target="--flag value"`.

## Direct torchrun fallbacks
- DDP: `PYTHONPATH=. torchrun --nproc_per_node 2 labs/train_distributed/ddp.py --mode optimized --compile`
- FSDP: `PYTHONPATH=. torchrun --nproc_per_node 2 labs/train_distributed/train_fsdp.py --mode optimized --float8 --sequence-length 512 --steps 2`
- ZeRO-2 example: `PYTHONPATH=. torchrun --nproc_per_node 2 labs/train_distributed/zero2.py --mode optimized --steps 3 --hidden-size 2048 --batch-size 4` (baseline prints a warning on single-GPU).
- Symmetric memory training: `PYTHONPATH=. torchrun --nproc_per_node 2 ch4/symmetric_memory_training_advanced.py --demo optimizer` (set `--disable-symmetric` for baseline).

## Quick sanity
```bash
python -m compileall labs/train_distributed
```
