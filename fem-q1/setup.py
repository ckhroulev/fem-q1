from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

extra_compile_args=["-O3", "-ffast-math"]

# Define extensions
Q12D = Extension("Q12D",
                 sources=["Q12D.pyx"],
                 include_dirs=[numpy.get_include()],
                 extra_compile_args=extra_compile_args)

Q13D = Extension("Q13D",
                 sources=["Q13D.pyx"],
                 include_dirs=[numpy.get_include()],
                 extra_compile_args=extra_compile_args)

setup(name = "q1_finite_elements",
      version = "0.0.1",
      description = "Q1 finite elements utility classes",
      author = "Constantine Khroulev",
      author_email = "ckhroulev@alaska.edu",
      url = "https://github.com/ckhroulev/finite-elements",
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Q12D, Q13D])

