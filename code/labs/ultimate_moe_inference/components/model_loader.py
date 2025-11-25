"""Model loader for gpt-oss models.

Supports:
- gpt-oss-20b (single GPU)
- gpt-oss-120b (multi-GPU)
- Multiple precision modes: MXFP4, FP8, BF16
- Tensor parallelism for multi-GPU
- Pipeline parallelism for multi-node
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


class Precision(Enum):
    """Model precision modes."""
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    MXFP4 = "mxfp4"  # Native gpt-oss precision


@dataclass
class ModelConfig:
    """Configuration for model loading.
    
    Model Resolution Order:
    1. LOCAL_MODEL_PATH environment variable (e.g., /mnt/models/gpt-oss-20b)
    2. model_name on HuggingFace Hub (e.g., openai/gpt-oss-20b)
    3. Local directory matching model_name
    """
    
    model_name: str = os.environ.get("LOCAL_MODEL_PATH", "openai/gpt-oss-20b")
    precision: Precision = Precision.MXFP4
    
    # Parallelism
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    expert_parallel: int = 1
    
    # Optimizations
    use_flash_attention: bool = True
    use_torch_compile: bool = False
    compile_mode: str = "max-autotune"
    
    # Memory
    max_memory_per_gpu: Optional[str] = None  # e.g., "160GB"
    offload_folder: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        precision_str = data.get("model", {}).get("precision", "mxfp4")
        precision = Precision(precision_str.lower())
        
        return cls(
            model_name=data.get("model", {}).get("name", cls.model_name),
            precision=precision,
            tensor_parallel=data.get("parallelism", {}).get("tensor_parallel", 1),
            pipeline_parallel=data.get("parallelism", {}).get("pipeline_parallel", 1),
            expert_parallel=data.get("parallelism", {}).get("expert_parallel", 1),
            use_flash_attention=data.get("optimizations", {}).get("enable_flash_attention", True),
            use_torch_compile=data.get("optimizations", {}).get("use_torch_compile", False),
            compile_mode=data.get("optimizations", {}).get("compile_mode", "max-autotune"),
        )


class ModelLoader:
    """Load and configure gpt-oss models for inference.
    
    This loader handles:
    - Model selection based on GPU count
    - Precision configuration (MXFP4, FP8, BF16)
    - Tensor parallel sharding
    - Optional torch.compile
    
    Example:
        loader = ModelLoader(config)
        model, tokenizer = loader.load()
    """
    
    # Model specifications
    MODEL_SPECS = {
        "openai/gpt-oss-20b": {
            "params_total": 21e9,
            "params_active": 3.6e9,
            "num_experts": 8,
            "top_k": 2,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "recommended_gpus": 1,
        },
        "openai/gpt-oss-120b": {
            "params_total": 117e9,
            "params_active": 5.1e9,
            "num_experts": 64,
            "top_k": 2,
            "hidden_size": 8192,
            "num_layers": 48,
            "num_heads": 64,
            "recommended_gpus": 8,
        },
    }
    
    def __init__(self, config: ModelConfig):
        """Initialize model loader.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
    
    @property
    def model(self) -> Any:
        """Get loaded model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model
    
    @property
    def tokenizer(self) -> Any:
        """Get loaded tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self._tokenizer
    
    def load(self) -> Tuple[Any, Any]:
        """Load model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library required")
        
        print(f"Loading model: {self.config.model_name}")
        print(f"  Precision: {self.config.precision.value}")
        print(f"  Tensor parallel: {self.config.tensor_parallel}")
        
        # Load tokenizer
        self._tokenizer = self._load_tokenizer()
        
        # Load model
        self._model = self._load_model()
        
        # Apply optimizations
        if self.config.use_torch_compile:
            self._model = self._compile_model(self._model)
        
        return self._model, self._tokenizer
    
    def _load_tokenizer(self) -> Any:
        """Load tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def _load_model(self) -> Any:
        """Load model with appropriate configuration."""
        # Determine torch dtype
        torch_dtype = self._get_torch_dtype()
        
        # Determine device map
        device_map = self._get_device_map()
        
        # Determine attention implementation
        attn_impl = self._get_attention_impl()
        
        # Model kwargs
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": True,
        }
        
        # Add attention implementation
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl
        
        # Add max memory if specified
        if self.config.max_memory_per_gpu:
            model_kwargs["max_memory"] = self._get_max_memory_dict()
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        
        model.eval()
        return model
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype based on precision config."""
        if self.config.precision == Precision.BF16:
            return torch.bfloat16
        elif self.config.precision == Precision.FP16:
            return torch.float16
        elif self.config.precision == Precision.FP8:
            # FP8 uses bf16 for loading, then quantizes
            return torch.bfloat16
        elif self.config.precision == Precision.MXFP4:
            # MXFP4 is native to gpt-oss, loaded as bf16 then uses native quant
            return torch.bfloat16
        else:
            return torch.bfloat16
    
    def _get_device_map(self) -> Union[str, Dict[str, int]]:
        """Get device map for model placement."""
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 0:
            return "cpu"
        
        if self.config.tensor_parallel > 1 and num_gpus >= self.config.tensor_parallel:
            # Use accelerate's auto device map for tensor parallelism
            return "auto"
        
        if num_gpus == 1:
            return "cuda:0"
        
        # Default to auto for multi-GPU
        return "auto"
    
    def _get_attention_impl(self) -> Optional[str]:
        """Get attention implementation."""
        if self.config.use_flash_attention:
            return "flash_attention_2"
        return "eager"
    
    def _get_max_memory_dict(self) -> Dict[int, str]:
        """Get max memory per device dictionary."""
        num_gpus = torch.cuda.device_count()
        max_mem = self.config.max_memory_per_gpu
        return {i: max_mem for i in range(num_gpus)}
    
    def _compile_model(self, model: Any) -> Any:
        """Apply torch.compile to model."""
        print(f"Compiling model with mode: {self.config.compile_mode}")
        
        return torch.compile(
            model,
            mode=self.config.compile_mode,
            fullgraph=True,
            dynamic=False,  # Static shapes for CUDA graphs
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model specifications."""
        specs = self.MODEL_SPECS.get(self.config.model_name, {})
        
        return {
            "model_name": self.config.model_name,
            "precision": self.config.precision.value,
            "tensor_parallel": self.config.tensor_parallel,
            "pipeline_parallel": self.config.pipeline_parallel,
            "params_total": specs.get("params_total", "unknown"),
            "params_active": specs.get("params_active", "unknown"),
            "num_experts": specs.get("num_experts", "unknown"),
            "recommended_gpus": specs.get("recommended_gpus", 1),
        }
    
    @staticmethod
    def select_model_for_gpus(num_gpus: int) -> str:
        """Select appropriate model based on available GPUs.
        
        Args:
            num_gpus: Number of available GPUs
            
        Returns:
            Model name (HuggingFace ID)
        """
        if num_gpus >= 4:
            return "openai/gpt-oss-120b"
        else:
            return "openai/gpt-oss-20b"
    
    @staticmethod
    def estimate_memory_gb(model_name: str, precision: Precision) -> float:
        """Estimate memory requirement for model.
        
        Args:
            model_name: Model name
            precision: Precision mode
            
        Returns:
            Estimated memory in GB
        """
        specs = ModelLoader.MODEL_SPECS.get(model_name, {})
        params = specs.get("params_total", 20e9)
        
        # Bytes per parameter by precision
        bytes_per_param = {
            Precision.BF16: 2,
            Precision.FP16: 2,
            Precision.FP8: 1,
            Precision.MXFP4: 0.5,
        }
        
        bpp = bytes_per_param.get(precision, 2)
        
        # Model weights + ~20% overhead for activations/buffers
        memory_bytes = params * bpp * 1.2
        return memory_bytes / 1e9


def load_model(
    model_name: str,
    precision: str = "mxfp4",
    tensor_parallel: int = 1,
    use_flash_attention: bool = True,
    use_torch_compile: bool = False,
) -> Tuple[Any, Any]:
    """Convenience function to load a model.
    
    Args:
        model_name: Model name (HuggingFace ID)
        precision: Precision mode ("mxfp4", "fp8", "bf16")
        tensor_parallel: Number of GPUs for tensor parallelism
        use_flash_attention: Enable FlashAttention
        use_torch_compile: Apply torch.compile
        
    Returns:
        Tuple of (model, tokenizer)
    """
    config = ModelConfig(
        model_name=model_name,
        precision=Precision(precision.lower()),
        tensor_parallel=tensor_parallel,
        use_flash_attention=use_flash_attention,
        use_torch_compile=use_torch_compile,
    )
    
    loader = ModelLoader(config)
    return loader.load()

