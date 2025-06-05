from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='ransac_voting',
    ext_modules=[
        CUDAExtension('ransac_voting', [
            './src/ransac_voting.cpp',
            './src/ransac_voting_kernel.cu'
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3']
        },
        include_dirs=[
            os.path.dirname(torch.__file__) + '/include',
            os.path.dirname(torch.__file__) + '/include/torch/csrc/api/include'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
