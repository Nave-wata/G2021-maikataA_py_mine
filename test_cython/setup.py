from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

ext = Extension("test_1", sources=["test_1.pyx"], include_dirs=['.', get_include()])
setup(name="test_1", ext_modules=cythonize([ext]))