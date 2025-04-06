import os
import sys
import subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'
    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    return cmd_out.stdout.decode('utf-8').strip()[:7]

def make_cuda_ext(name, module, sources, include_dirs=None, library_dirs=None, libraries=None, extra_compile_args=None):
    return CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources],
        include_dirs=include_dirs or [],
        library_dirs=library_dirs or [],
        libraries=libraries or [],
        extra_compile_args=extra_compile_args or {'nvcc': ['-Xcompiler', '-fPIC']}
    )

def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)

if __name__ == '__main__':
    version = '0.6.0+%s' % get_git_commit_number()
    write_version_to_file(version, 'pcdet/version.py')

    cuda_include_dir = '/usr/local/cuda-11.8/include'
    cuda_lib_dir = '/usr/local/cuda-11.8/lib64'
    if not os.path.isdir(cuda_include_dir) or not os.path.isdir(cuda_lib_dir):
        raise RuntimeError(f"CUDA paths not found. Check paths: {cuda_include_dir}, {cuda_lib_dir}")

    if 'nv3d' in sys.argv:
        setup(
            name='pcdet_nv3d',
            version=version,
            ext_modules=[
                make_cuda_ext(
                    name='nv3d_cuda',
                    module='pcdet.ops.nv3d',
                    sources=[
                        'src/nv3d_api.cpp',
                        'src/compute_normals_kernel.cu',
                        'src/compute_normals_gpu.cpp',
                        'src/compute_seven_nn_kernel.cu',
                        'src/compute_seven_nn_gpu.cpp',
                        'src/compute_density_kernel.cu',
                        'src/compute_density_gpu.cpp',
                        'src/compute_norm_mask_kernel.cu',
                        'src/compute_norm_mask_gpu.cpp'
                    ],
                    library_dirs=[cuda_lib_dir],
                    libraries=['cublas', 'cusolver', 'curand']
                )
            ],
            cmdclass={'build_ext': BuildExtension}
        )

    elif 'main' in sys.argv:
        setup(
            name='pcdet',
            version=version,
            description='OpenPCDet is a general codebase for 3D object detection from point cloud',
            install_requires=[
                'numpy',
                'llvmlite',
                'numba',
                'tensorboardX',
                'easydict',
                'pyyaml',
                'scikit-image',
                'tqdm',
                'SharedArray',
                # 'spconv',  # spconv has different names depending on the cuda version
            ],

            author='Shaoshuai Shi',
            author_email='shaoshuaics@gmail.com',
            license='Apache License 2.0',
            packages=find_packages(exclude=['tools', 'data', 'output']),
            cmdclass={
                'build_ext': BuildExtension,
            },
            ext_modules=[
                make_cuda_ext(
                    name='iou3d_nms_cuda',
                    module='pcdet.ops.iou3d_nms',
                    sources=[
                        'src/iou3d_cpu.cpp',
                        'src/iou3d_nms_api.cpp',
                        'src/iou3d_nms.cpp',
                        'src/iou3d_nms_kernel.cu',
                    ]
                ),
                make_cuda_ext(
                    name='roiaware_pool3d_cuda',
                    module='pcdet.ops.roiaware_pool3d',
                    sources=[
                        'src/roiaware_pool3d.cpp',
                        'src/roiaware_pool3d_kernel.cu',
                    ]
                ),
                make_cuda_ext(
                    name='roipoint_pool3d_cuda',
                    module='pcdet.ops.roipoint_pool3d',
                    sources=[
                        'src/roipoint_pool3d.cpp',
                        'src/roipoint_pool3d_kernel.cu',
                    ]
                ),
                make_cuda_ext(
                    name='pointnet2_stack_cuda',
                    module='pcdet.ops.pointnet2.pointnet2_stack',
                    sources=[
                        'src/pointnet2_api.cpp',
                        'src/ball_query.cpp',
                        'src/ball_query_gpu.cu',
                        'src/group_points.cpp',
                        'src/group_points_gpu.cu',
                        'src/sampling.cpp',
                        'src/sampling_gpu.cu', 
                        'src/interpolate.cpp', 
                        'src/interpolate_gpu.cu',
                        'src/voxel_query.cpp', 
                        'src/voxel_query_gpu.cu',
                        'src/vector_pool.cpp',
                        'src/vector_pool_gpu.cu'
                    ],
                ),
                make_cuda_ext(
                    name='pointnet2_batch_cuda',
                    module='pcdet.ops.pointnet2.pointnet2_batch',
                    sources=[
                        'src/pointnet2_api.cpp',
                        'src/ball_query.cpp',
                        'src/ball_query_gpu.cu',
                        'src/group_points.cpp',
                        'src/group_points_gpu.cu',
                        'src/interpolate.cpp',
                        'src/interpolate_gpu.cu',
                        'src/sampling.cpp',
                        'src/sampling_gpu.cu',

                    ],
                ),
                make_cuda_ext(
                    name="bev_pool_ext",
                    module="pcdet.ops.bev_pool",
                    sources=[
                        "src/bev_pool.cpp",
                        "src/bev_pool_cuda.cu",
                    ],
                ),
                make_cuda_ext(
                    name='ingroup_inds_cuda',
                    module='pcdet.ops.ingroup_inds',
                    sources=[
                        'src/ingroup_inds.cpp',
                        'src/ingroup_inds_kernel.cu',
                    ]
                ),
            ],
        )