from setuptools import setup, Extension
import pybind11
import os

# Define the extension module
bullet_module = Extension(
    'bullet_bindings',
    sources=['bullet_python_bindings.cpp'],
    include_dirs=[
        pybind11.get_include(),
        '../' # For bullet-core.cpp
    ],
    extra_compile_args=['-std=c++17', '-O3'],
    language='c++'
)

setup(
    name='bullet-ai',
    version='1.0.2',
    description='Bullet OS Python Bindings',
    ext_modules=[bullet_module],
    py_modules=['bullet_py_api'],
    install_requires=['pybind11'],
    zip_safe=False,
)
