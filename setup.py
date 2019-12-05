from setuptools import setup, Extension
from distutils.command.build_ext import build_ext
from distutils.sysconfig import customize_compiler
from distutils import ccompiler
import glob
import shutil
import os
import subprocess
import numpy

# Hide all the horrible parts away.
from setup_functions import get_starlink_macros, get_source, \
    setup_building

"""
Setup script for the HDS and NDF python extensions.
"""


# First we set up the files for building. This is safe to repeat again
# and again.

# Create a custom build script for the extension.

# variable to hold the list of dependencies for HDS.
HDS_DEP_LIBS = ('starutil', 'starmem', 'cnf', 'ems', 'mers',
                'chr', 'hds-v4', 'hds-v5', 'one')
HDS_DEP_INCLUDES = ('include/', 'missingincludes/') + HDS_DEP_LIBS + \
                   ('hds-v4_missingincludes', 'hds-v5_missingincludes', 'hdf5/src/', 'hdf5/hl/src')

class custom_star_build(build_ext):

    """
    Custom build for Starlink's HDS lib

    This will build HDF5 (using configure/make) and also all of the
    starilnk requirements for building HDS (using the c compiler.

    It will add the results of those builds to every extension in
    self.extensions, and will then call the standard build_ext.run
    command to finish building the extensions.

    To also build NDF, it will need the source ot its own additional
    dependencies and the code for HDS to be included in its source
    lists. The result is only HDS itself will have to be built twice.
    """
    def run(self):
        # Ensure the directories and files are in appropriate locations.
        setup_building()

        # Before we can build the extensions, we have to run ./configure
        # and make for HDF5.
        basedir = os.getcwd()
        os.chdir('hdf5')
        env = os.environ
        subprocess.check_call('./configure', env=env)
        subprocess.check_call('make', env=env)
        os.chdir(basedir)

        #Now we need to get the header files we need copied into our
        #output directory.
        if not os.path.isdir('include'):
            os.mkdir('include')
        if not os.path.isdir(os.path.join('include', 'star')):
            os.mkdir(os.path.join('include', 'star'))

        # Copy hdf5 needed header files over.
        shutil.copy('hdf5/src/hdf5.h', 'include')
        shutil.copy('hdf5/src/H5public.h', 'include')
        shutil.copy('hdf5/src/H5pubconf.h', 'include')
        shutil.copy('hdf5/src/H5version.h', 'include')


        # Get the built object files for later.
        hdf5_extras = glob.glob('hdf5/src/.libs/*.o')

        # now build the HDS dependency packages. Can we just do
        # this all together?
        hds_source_dep = []
        for name_ in HDS_DEP_LIBS:
            hds_source_dep += get_source(name_)
        define_macros = get_starlink_macros()

        # Now build all.
        outputdir= 'temp'
        compiler = ccompiler.new_compiler()

        # use distutils customize compiler to fix this.
        customize_compiler(compiler)
        extraobjs = compiler.compile(sources=hds_source_dep, output_dir=outputdir,
                                     macros=define_macros, include_dirs=HDS_DEP_INCLUDES,
                                     )


        for ext in self.extensions:
            ext.extra_objects += hdf5_extras
            ext.extra_objects += extraobjs

        build_ext.run(self)


# Get the Starlink specific defines.
defines = get_starlink_macros()

# Check for ast: ensure it is present in all builds being done for pypi.
# TODO: ensure at run time library checks for this instead!
try:
    from starlink import Ast
    defines.append(('HAVE_AST', '1'))
    have_ast = True
except ImportError:
    have_ast = False
    print("")
    print("  Will not be building with AST facilities.")
    print("  Install starlink.Ast in order to read and write AST FrameSets.")
    print("")


# Get the lists of source files for the NDF extra dependencies.
ndf_source_dep = []
for name_ in ['prm', 'ast', 'ary']:
    ndf_source_dep += get_source(name_)



hdsex_includedirs = ['include/', 'hds/', 'missingincludes/',
                     'hds_missingincludes/', 'hdf5/src/', 'hdf5/hl/src'] + \
    ['starutil', 'starmem/', 'cnf', 'ems', 'mers', 'chr', 'hds-v4', 'hds-v5', 'one'] + \
    [numpy.get_include()]


ndfex_includedirs = hdsex_includedirs + ['prm', 'ast', 'ary', 'ast_missingincludes/']

# Can't build NDF without Ast!
if have_ast:
    ndfex_includedirs.append(Ast.get_include())

# Define the two extensions.
pythonhds_source = [os.path.join('starlink', 'hds', 'hds.c')]

hdsExtension = Extension('starlink.hds',
                         sources = get_source('hds') + pythonhds_source,
                         include_dirs = hdsex_includedirs,
                         define_macros = defines,
                         libraries = ['z'],
)

pythonndf_source = [os.path.join('starlink', 'ndf', 'ndf.c')]

ndfExtension = Extension('starlink.ndf',
                         sources = get_source('hds') + ndf_source_dep + get_source('ndf') + pythonndf_source,
                         include_dirs = ['ndf/', 'ndf_missingincludes/'] + ndfex_includedirs,
                         define_macros = defines,
                         libraries = ['z'],
)






with open('README.rst') as file:
    long_description = file.read()



setup(name='starlink-pyndf',
      version='0.3',
      long_description=long_description,
      packages=['starlink'],#, 'starlink.ndfpack'],
      cmdclass={'build_ext': custom_star_build},
      ext_modules=[hdsExtension, ndfExtension],
      test_suite='test',

      # metdata
      author='SF Graves',
      author_email='s.graves@eaobservatory.org',
      url='https://github.com/sfgraves/starlink-pyhds',
      license="GNU GPL v3",
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Programming Language :: Python',
          'Programming Language :: C',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      install_requires = ['numpy',],

)
