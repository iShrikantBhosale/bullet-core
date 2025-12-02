from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build bullet_core")
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Get pybind11 cmake path
        import pybind11
        pybind11_cmake = pybind11.get_cmake_dir()
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-Dpybind11_DIR={pybind11_cmake}'
        ]
        
        build_args = ['--config', 'Release', '--', '-j4']
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bullet_core",
    version="0.1.0",
    author="Bullet OS Team",
    author_email="",
    description="Bullet-Core: SIMD-optimized CPU kernels for deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension('bullet_core_cpp', sourcedir='.')],
    cmdclass={'build_ext': CMakeBuild},
    packages=['bullet_core'],
    package_dir={'bullet_core': 'python'},
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
