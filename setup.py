from distutils.core import setup
from Cython.Build import cythonize

# COMPILE WITH :  python setup.py build_ext --inplace

setup( ext_modules = cythonize("cpython_polymul.pyx") )

