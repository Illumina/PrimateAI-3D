
import platform
import sys
from setuptools import setup

from distutils.core import Extension
from Cython.Build import cythonize

EXTRA_COMPILE_ARGS = ["-std=c++11"]
EXTRA_LINK_ARGS = []
if sys.platform == "darwin":
    EXTRA_COMPILE_ARGS += [
        "-stdlib=libc++",
        "-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1",
        "-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include",
        ]
    EXTRA_LINK_ARGS += [
        "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib",
        ]
elif sys.platform == "win32":
    EXTRA_COMPILE_ARGS += ['/std:c++14']

if platform.machine() == 'x86_64':
    EXTRA_COMPILE_ARGS += ['-mavx', '-mavx2', '-mfma']

ext = cythonize([
    Extension("rvPRS.rare.id_decoder",
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        sources=[
            "rvPRS/rare/decode_ids.pyx",
            "src/convert_ids.cpp",
            ],
        include_dirs=["src/"],
        language="c++"),
    Extension('rvPRS.rare.variant',
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        sources=[
            "rvPRS/rare/variant.pyx",
            "src/convert_ids.cpp",
            ],
        include_dirs=["src/"],
        language="c++"),
    ]
)

setup(name="rvPRS",
    description='Package for rare variant polygenic scores',
    version="1.0.0",
    author="Jeremy McRae",
    author_email="jmcrae@illumina.com",
    url='https://github.com/illumina/PrimateAI-3D',
    packages=["rvPRS", "rvPRS.common", "rvPRS.rare", "rvPRS.rare.bin"],
    package_data={},
    entry_points={'console_scripts': [
        'rvPRS = rvPRS.rare.bin.rare_prs:main',
        ]},
    license='GPLv3',
    install_requires=[
        'scipy',
        'numpy',
        'gencodegenes',
        'bgen',
        'scikit-learn',
        'regressor',
        'fisher',
    ],
    ext_modules=ext,
    test_suite='unittest:TestLoader',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ])
