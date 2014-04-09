from distutils.core import setup, Extension

module1 = Extension('grabcutobf',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    include_dirs = [],
                    libraries = ['opencv_core', 'opencv_highgui', 'opencv_imgproc'],
                    library_dirs = [],
                    sources = ['grabcutobfmodule.cpp', 'grabcutobf.cpp'],
                    extra_compile_args = ['-Wno-unused-function'])

setup (name = 'PackageGrabCutObf',
       version = '0.1',
       description = 'This is the package for grabcutobf',
       author = 'Alessandro Bergamo',
       ext_modules = [module1])
