from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='geotransformer',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='geotransformer.ext',
            sources=[
                'geotransformer/extra/cloud/cloud.cpp',
                'geotransformer/cpu/grid_subsampling/grid_subsampling.cpp',
                'geotransformer/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'geotransformer/cpu/radius_neighbors/radius_neighbors.cpp',
                'geotransformer/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'geotransformer/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
