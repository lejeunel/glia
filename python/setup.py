#!/usr/bin/env python

from __future__ import division
#from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import os, sys
from os.path import join

# We require Python v2.7 or newer
if sys.version_info[:2] < (2,7): raise RuntimeError("This requires Python v2.7 or newer")

# Prepare for compiling the source code
from distutils.ccompiler import get_default_compiler
import numpy
compiler_name = get_default_compiler() # TODO: this isn't the compiler that will necessarily be used, but is a good guess...
compiler_opt = {
    'msvc'    : ['/D_SCL_SECURE_NO_WARNINGS','/EHsc','/O2','/DNPY_NO_DEPRECATED_API=7','/bigobj','/openmp'],
    # TODO: older versions of gcc need -std=c++0x instead of -std=c++11
    'unix'    : ['-std=c++11','-O3','-DNPY_NO_DEPRECATED_API=7','-fopenmp'], # gcc/clang (whatever is system default)
    'mingw32' : ['-std=c++11','-O3','-DNPY_NO_DEPRECATED_API=7','-fopenmp'],
    'cygwin'  : ['-std=c++11','-O3','-DNPY_NO_DEPRECATED_API=7','-fopenmp'],
}
linker_opt = {
    'msvc'    : [],
    'unix'    : ['-fopenmp'], # gcc/clang (whatever is system default)
    'mingw32' : ['-fopenmp'],
    'cygwin'  : ['-fopenmp'],
}
np_inc = numpy.get_include()
import pysegtools
cy_inc = join(os.path.dirname(pysegtools.__file__), 'general', 'cython') # TODO: better way to get this
src_ext = '.cpp'
def create_ext(name, dep=[], src=[], inc=[], lib=[], objs=[]):
    from distutils.extension import Extension
    return Extension(
        name=name,
        depends=dep,
        sources=[join(*name.split('.'))+src_ext]+src,
        define_macros=[('NPY_NO_DEPRECATED_API','7'),],
        include_dirs=[np_inc,cy_inc]+inc,
        library_dirs=lib,
        extra_objects=objs,
        extra_compile_args=compiler_opt.get(compiler_name, []),
        extra_link_args=linker_opt.get(compiler_name, []),
        language='c++',
    )

# Find and use Cython if available
try:
    from distutils.version import StrictVersion
    import Cython.Build
    if StrictVersion(Cython.__version__) >= StrictVersion('0.22'):
        src_ext = '.pyx'
        def cythonize(*args, **kwargs):
            kwargs.setdefault('include_path', []).append(cy_inc)
            return Cython.Build.cythonize(*args, **kwargs)
except ImportError:
    def cythonize(exts, *args, **kwargs): return exts

# Finally we get to run setup
try: from setuptools import setup
except ImportError: from distutils.core import setup
setup(name='glia',
      version='0.1',
      author='Jeffrey Bush',
      author_email='jeff@coderforlife.com',
      packages=['glia'],
      setup_requires=['numpy>=1.7'],
      install_requires=['numpy>=1.7','scipy>=0.16','pysegtools>=0.1'],
      use_2to3=True, # the code *should* support Python 3 once run through 2to3 but this isn't tested
      zip_safe=False,
      package_data = { '': ['*.pyx', '*.pyxdep', '*.pxi', '*.pxd', '*.h', '*.txt'], }, # Make sure all Cython files are wrapped up with the code
      ext_modules = cythonize([
          create_ext('glia.__contours_around_labels'),
          create_ext('glia.__pairwise_pixel_values'),
      ])
)
