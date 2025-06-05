from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='nn_utils',
    ext_modules=[
        CUDAExtension('_ext', [
            'src/nearest_neighborhood.cu'
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3']
        },
        include_dirs=[
            os.path.dirname(os.path.abspath(__file__)) + '/src'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
