"""Mixture-of-Experts parallelism lab helpers."""

from .spec_helper import configure_spec_from_cli, is_plan_available, get_plan_error

configure_spec_from_cli()

# Only import plan components if available
if is_plan_available():
    from labs.moe_parallelism.plan import (
        AVAILABLE_SPEC_PRESETS,
        ClusterSpec,
        ModelSpec,
        ParallelismPlan,
        PlanEvaluator,
        DEFAULT_CLUSTER,
        DEFAULT_MODEL,
        format_report,
        get_default_cluster_spec,
        get_default_model_spec,
        resolve_specs,
    )

    __all__ = [
        "ClusterSpec",
        "ModelSpec",
        "ParallelismPlan",
        "PlanEvaluator",
        "format_report",
        "DEFAULT_CLUSTER",
        "DEFAULT_MODEL",
        "AVAILABLE_SPEC_PRESETS",
        "get_default_cluster_spec",
        "get_default_model_spec",
        "resolve_specs",
        "is_plan_available",
        "get_plan_error",
    ]
else:
    # Plan not available - export only helper functions
    __all__ = ["is_plan_available", "get_plan_error"]
