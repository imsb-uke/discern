"""Build file for cython extensions."""
from distutils.core import Extension
from Cython.Build import cythonize

_EXTENSIONS = [
    Extension("discern.mmd._mmd",
              sources=["discern/mmd/_mmd.pyx", "discern/mmd/_cmmd.c"],
              language="c",
              extra_compile_args=["-O3"],
              libraries=["m"],
              include_dirs=["discern/mmd/"]),
]


def build(setup_kwargs):
    """This function is mandatory in order to build the extensions."""
    setup_kwargs.update({'ext_modules': cythonize(_EXTENSIONS)})
