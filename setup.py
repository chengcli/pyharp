"""Setup configuration for Python packaging."""
# pylint: disable = deprecated-module, exec-used
import os
import sys
import platform
import glob
from pathlib import Path
from setuptools import setup
from torch.utils import cpp_extension
import torch

def parse_library_names(libdir):
    """Parse the library files."""
    # Get the library files
    library_files = []
    for root, _, files in os.walk(libdir):
        for file in files:
            if file.endswith('.a') or file.endswith('.so'):
                library_files.append(os.path.join(root, file))

    # Extract the library names from the library files
    library_names = []
    for library_file in library_files:
        file_name = os.path.basename(library_file)
        # remove lib and .so or .a
        library_name = file_name[3:].rsplit('.', 1)[0]
        library_names.append(library_name)

    # add homebrew libraries if on MacOS
    if platform.system() == 'Darwin':
        library_names.extend(['netcdf'])

    return library_names

def check_requirements():
    """Check if the system requirements are met."""
    # Check the operating system
    os_name = platform.system()
    if os_name not in ['Darwin', 'Linux']:
        sys.stderr.write(
            "Unsupported operating system. Please use MacOS or Linux.\n")
        return False

    # Min python version is Python3.8
    if sys.version_info < (3, 8):
        sys.stderr.write("Python 3.8 or higher is required.\n")
        return False

    return True

# If the system does not meet requirement, exit.
if not check_requirements():
    sys.exit(1)

# Setup configuration
current_dir = os.getenv('WORKSPACE')

if not current_dir:
    current_dir = Path().absolute()

# add homebrew directories if on MacOS
if platform.system() == 'Darwin':
    extra_libdirs = ['/opt/homebrew/lib']
else:
    extra_libdirs = []

if torch.cuda.is_available():
    setup(
        package_dir={"pyharp": "python"},
        packages=["pyharp"],
        ext_modules=[cpp_extension.CUDAExtension(
            name = 'pyharp.pyharp',
            sources = glob.glob('python/csrc/*.cpp')
            + glob.glob('src/**/*.cu', recursive=True),
            include_dirs = [
                f'{current_dir}',
                f'{current_dir}/build',
                f'{current_dir}/build/_deps/fmt-src/include'
            ],
            library_dirs = [f'{current_dir}/build/lib'] + extra_libdirs,
            libraries = parse_library_names(f'{current_dir}/build/lib'),
            extra_compile_args = {'nvcc': ['--extended-lambda']},
            )],
        cmdclass={'build_ext': cpp_extension.BuildExtension},
    )
else:
    setup(
        package_dir={"pyharp": "python"},
        packages=["pyharp"],
        ext_modules=[cpp_extension.CppExtension(
            name = 'pyharp.pyharp',
            sources = glob.glob('python/csrc/*.cpp'),
            include_dirs = [
                f'{current_dir}',
                f'{current_dir}/build',
                f'{current_dir}/build/_deps/fmt-src/include',
                f'{current_dir}/build/_deps/disort-src',
                f'{current_dir}/build/_deps/yaml-cpp-src/include'
            ],
            library_dirs = [f'{current_dir}/build/lib'] + extra_libdirs,
            libraries = parse_library_names(f'{current_dir}/build/lib'),
            extra_link_args=[""]
            )],
        cmdclass={'build_ext': cpp_extension.BuildExtension},
    )
