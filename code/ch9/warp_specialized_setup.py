"""Setup script for warp specialization CUDA extension."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='warp_specialized_cuda',
    ext_modules=[
        CUDAExtension('warp_specialized_cuda', [
            'warp_specialized_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

