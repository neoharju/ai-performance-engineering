#!/usr/bin/env python3
"""Optimized MoE with BMM Fusion.

This wrapper runs Level 5 (BMM Fusion) for the aisp bench CLI.
"""

from labs.moe_optimization_journey.level5_bmm_fusion import Level5BMMFusion

def get_benchmark():
    return Level5BMMFusion()

if __name__ == "__main__":
    Level5BMMFusion.main()




