from __future__ import print_function
from distutils.core import setup, Extension
from os.path import abspath, join, dirname
import sys
import os, os.path
from numpy import get_include

numpy_include_dir = get_include()

# Version reporting/recording
# The ABSTRACT_RENDERING_VERSION is read from _version.txt whenver setup is run
# This version is also used in pypi.
# If the file is not found, it is created with version -1.-1.-1

if not os.path.isfile('abstract_rendering/_version.txt'):
    with open('abstract_rendering/_version.txt', 'w') as f:
        f.write('-1.-1.-1')

with open('abstract_rendering/_version.txt', 'r') as f:
    ABSTRACT_RENDERING_VERSION = f.read().strip()


# Flag to indicate that libdispatch should be used (OS X only)
DISPATCH_FLAG = "--dispatch"

if DISPATCH_FLAG in sys.argv:
    transform = Extension('abstract_rendering.transform_libdispatch',
            ['abstract_rendering/transform_libdispatch.cpp'],
            extra_compile_args=['-O3', '-Wall', '-fno-rtti', '-fno-exceptions', '-fPIC', '-lstdc++'])
    del sys.argv[sys.argv.index(DISPATCH_FLAG)]

else:
    transform = Extension('abstract_rendering.transform',
            ['abstract_rendering/transform.cpp'],
            extra_compile_args=['-O3', '-Wall', '-fno-rtti', '-fno-exceptions', '-fPIC', '-lstdc++'])


def getsitepackages():
    """Returns a list containing all global site-packages directories
    (and possibly site-python)."""

    _is_64bit = (getattr(sys, 'maxsize', None) or getattr(sys, 'maxint')) > 2**32
    _is_pypy = hasattr(sys, 'pypy_version_info')
    _is_jython = sys.platform[:4] == 'java'

    prefixes = [sys.prefix, sys.exec_prefix]

    sitepackages = []
    seen = set()

    for prefix in prefixes:
        if not prefix or prefix in seen:
            continue
        seen.add(prefix)

        if sys.platform in ('os2emx', 'riscos') or _is_jython:
            sitedirs = [os.path.join(prefix, "Lib", "site-packages")]
        elif _is_pypy:
            sitedirs = [os.path.join(prefix, 'site-packages')]
        elif sys.platform == 'darwin' and prefix == sys.prefix:
            if prefix.startswith("/System/Library/Frameworks/"):   # Apple's Python
                sitedirs = [os.path.join("/Library/Python", sys.version[:3], "site-packages"),
                            os.path.join(prefix, "Extras", "lib", "python")]

            else:  # any other Python distros on OSX work this way
                sitedirs = [os.path.join(prefix, "lib",
                            "python" + sys.version[:3], "site-packages")]

        elif os.sep == '/':
            sitedirs = [os.path.join(prefix,
                                     "lib",
                                     "python" + sys.version[:3],
                                     "site-packages"),
                        os.path.join(prefix, "lib", "site-python"),
                        os.path.join(prefix, "python" + sys.version[:3], "lib-dynload")]
            lib64_dir = os.path.join(prefix, "lib64", "python" + sys.version[:3], "site-packages")
            if (os.path.exists(lib64_dir) and
                os.path.realpath(lib64_dir) not in [os.path.realpath(p) for p in sitedirs]):
                if _is_64bit:
                    sitedirs.insert(0, lib64_dir)
                else:
                    sitedirs.append(lib64_dir)
            try:
                # sys.getobjects only available in --with-pydebug build
                sys.getobjects
                sitedirs.insert(0, os.path.join(sitedirs[0], 'debug'))
            except AttributeError:
                pass
            # Debian-specific dist-packages directories:
            if sys.version[0] == '2':
                sitedirs.append(os.path.join(prefix, "lib",
                                             "python" + sys.version[:3],
                                             "dist-packages"))
            else:
                sitedirs.append(os.path.join(prefix, "lib",
                                             "python" + sys.version[0],
                                             "dist-packages"))
            sitedirs.append(os.path.join(prefix, "local/lib",
                                         "python" + sys.version[:3],
                                         "dist-packages"))
            sitedirs.append(os.path.join(prefix, "lib", "dist-python"))
        else:
            sitedirs = [prefix, os.path.join(prefix, "lib", "site-packages")]
        if sys.platform == 'darwin':
            # for framework builds *only* we add the standard Apple
            # locations. Currently only per-user, but /Library and
            # /Network/Library could be added too
            if 'Python.framework' in prefix:
                home = os.environ.get('HOME')
                if home:
                    sitedirs.append(
                        os.path.join(home,
                                     'Library',
                                     'Python',
                                     sys.version[:3],
                                     'site-packages'))
        for sitedir in sitedirs:
            sitepackages.append(os.path.abspath(sitedir))
    return sitepackages


site_packages = getsitepackages()[0]
old_dir = join(site_packages, "abstract_rendering")
path_file = join(site_packages, "abstract_rendering.pth")
path = abspath(dirname(__file__))

if 'develop' in sys.argv:
    print("Develop mode.")
    if os.path.isdir(old_dir):
        print("  - Removing package %s." % old_dir)
        import shutil
        shutil.rmtree(old_dir)
    with open(path_file, "w+") as f:
        f.write(path)
    print("  - writing path '%s' to %s" % (path, path_file))
    print()
    sys.exit()

setup(name='abstract_rendering',
      version= ABSTRACT_RENDERING_VERSION,
      description='Rendering as a binning process',
      author='Joseph Cottam',
      author_email='jcottam@indiana.edu',
      url='https://github.com/JosephCottam/AbstractRendering',
      packages=['abstract_rendering', 'abstract_rendering.test'],
      package_dir={'abstract_rendering': 'abstract_rendering'},
      package_data={'abstract_rendering': ['*.txt', '*.so', '*.dll']},
      ext_modules=[
          transform,
          Extension('abstract_rendering._cntr',
              ['abstract_rendering/cntr.c'],
              include_dirs=[numpy_include_dir],
              define_macros=[('NUMPY', None)],
              extra_compile_args=['-O3', '-Wall', '-fno-rtti', '-fno-exceptions', '-fPIC'])
          ])
