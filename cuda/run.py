from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='emd',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Earth Mover\'s Distance implementation using PyTorch and CUDA',
    long_description='A PyTorch implementation of the Earth Mover\'s Distance (EMD) algorithm using CUDA for GPU acceleration.',
    url='https://github.com/your_username/emd',
    license='MIT',
    keywords='pytorch cuda emd earth mover\'s distance',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    ext_modules=[
        CUDAExtension('emd', [
            'emd.cpp',
            'emd_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='emd',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Earth Mover\'s Distance implementation using PyTorch and CUDA',
    long_description='A PyTorch implementation of the Earth Mover\'s Distance (EMD) algorithm using CUDA for GPU acceleration.',
    url='https://github.com/your_username/emd',
    license='MIT',
    keywords='pytorch cuda emd earth mover\'s distance',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    ext_modules=[
        CUDAExtension('emd', [
            'emd.cpp',
            'emd_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
