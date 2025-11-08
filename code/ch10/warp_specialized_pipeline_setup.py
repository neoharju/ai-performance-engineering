"""Setup script for warp specialization CUDA extension (Chapter 10)."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='warp_specialized_pipeline_enhanced',
    ext_modules=[
        CUDAExtension('warp_specialized_pipeline_enhanced', [
            'warp_specialized_pipeline_enhanced_extension.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

