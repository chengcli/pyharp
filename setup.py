#!/usr/bin/env python
import os
import glob
import torch
import platform
from pathlib import Path
from setuptools import setup
from torch.utils import cpp_extension

def parse_library_names(libdir):
    library_names = []
    for root, _, files in os.walk(libdir):
        for file in files:
            if file.endswith((".a", ".so")):
                file_name = os.path.basename(file)
                library_names.append(file_name[3:].rsplit(".", 1)[0])

    # add system netcdf library
    library_names.extend(['netcdf'])

    return sorted(library_names)

current_dir = os.getenv("WORKSPACE", Path().absolute())
include_dirs = [
    f"{current_dir}",
    f"{current_dir}/build",
    f"{current_dir}/build/_deps/fmt-src/include",
    f'{current_dir}/build/_deps/pydisort-src',
    f'{current_dir}/build/_deps/yaml-cpp-src/include',
]

# add homebrew directories if on MacOS
lib_dirs = [f"{current_dir}/build/lib"]
if platform.system() == 'Darwin':
    lib_dirs.extend(['/opt/homebrew/lib'])
else:
    lib_dirs.extend(['/lib64/', '/usr/lib/x86_64-linux-gnu/'])

libraries = parse_library_names(f"{current_dir}/build/lib")

extra_link_args = []
#if platform.system() == 'Linux':
#    extra_link_args.extend(['-Wl,--whole-archive',
#                            f"{current_dir}/build/lib/libharp_release.a", '-Wl,--no-whole-archive'])

if torch.cuda.is_available():
    ext_module = cpp_extension.CUDAExtension(
        name='pyharp.pyharp',
        sources=glob.glob('python/csrc/*.cpp')
        + glob.glob('src/**/*.cu', recursive=True),
        include_dirs=include_dirs,
        library_dirs=lib_dirs,
        libraries=libraries,
        extra_compile_args={'nvcc': ['--extended-lambda'],
                            'cc': ["-Wno-attributes"]},
        extra_link_args=extra_link_args,
    )
else:
    ext_module = cpp_extension.CppExtension(
        name='pyharp.pyharp',
        sources=glob.glob('python/csrc/*.cpp'),
        include_dirs=include_dirs,
        library_dirs=lib_dirs,
        libraries=libraries,
        extra_compile_args=['-Wno-attributes'],
        extra_link_args=extra_link_args,
        )

setup(
    package_dir={"pyharp": "python"},
    packages=["pyharp"],
    ext_modules=[ext_module],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
