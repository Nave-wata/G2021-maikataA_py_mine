from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

ext = Extension("noise_detection_AI/test_2", sources=["noise_detection_AI/test_2.pyx"], include_dirs=['.', get_include()])
setup(name="noise_detection_AI/test_2", ext_modules=cythonize([ext]))