from setuptools import setup, Extension
from distutils.command.build_ext import build_ext
from distutils.sysconfig import customize_compiler
from distutils import ccompiler
import glob
import shutil
import os
import subprocess
import numpy
from Cython.Build import cythonize
from starlink import Ast

# Hide all the horrible parts away.
from setup_functions import get_starlink_macros, get_source, \
    setup_building

"""
Setup script for the HDS and NDF python extensions.
"""


# For development purposes: set to True to not actually rebuild hdf5
# and the individual libraries.
FAKEBUILDING=False
OUTPUTDIR= '.libs'

# variable to hold the list of dependencies for HDS.
HDS_DEP_LIBS = ('starutil', 'starmem', 'cnf', 'ems', 'mers',
                'chr', 'one')
HDS_DEP_INCLUDES = ('include/', 'missingincludes/') + HDS_DEP_LIBS + \
                   ('hdf5/src/', 'hdf5/hl/src')

# Create a custom build script for the extension.
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

        # Get the names of library files Name of shared object
        # library, and ensure -bundle isn't called in linking on osx.
        libtype = 'shared'
        rdirs = ['$ORIGIN']
        ldirs = []
        ext_runtime_library_dirs = '$ORIGIN/{}'
        ext_extra_link_args = None
        osx = None
        if 'osx' in self.plat_name or 'darwin' in self.plat_name:
            osx=True
            print('\n\nBuilding under OSX!\n\n\n')
            libtype = 'dylib'
            from distutils import sysconfig
            vars = sysconfig.get_config_vars()
            vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-dynamiclib')
            rdirs = []
            ldirs=['-Wl,-rpath,'+'@loader_path/']
            ldirs=[]
            ext_runtime_library_dirs = None
            ext_extra_link_args = '-Wl,-rpath,@loader_path/{}'
            install_name_pattern = '-Wl,-install_name,@rpath/{}'
        # Ensure the directories and files are in appropriate locations.
        setup_building()

        # Before we can build the extensions, we have to run ./configure
        # and make for HDF5.
        basedir = os.getcwd()
        os.chdir('hdf5')
        env = os.environ
        if not FAKEBUILDING:
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
        shutil.copy(os.path.join('hdf5','src','hdf5.h'), 'include')
        shutil.copy(os.path.join('hdf5','src','H5public.h'), 'include')
        shutil.copy(os.path.join('hdf5','src','H5pubconf.h'), 'include')
        shutil.copy(os.path.join('hdf5','src','H5version.h'), 'include')

        # Get the sources for the ndf and hds dependencies.
        hds_source_dep = []
        for name_ in HDS_DEP_LIBS:
            hds_source_dep += get_source(name_)

        ndf_source_dep = []
        for name_ in ['prm', 'ast', 'ary']:
            ndf_source_dep += get_source(name_)


        hdsex_includedirs = ['include', 'hds', 'missingincludes',
                             'hds_missingincludes', os.path.join('hdf5','src'), os.path.join('hdf5','hl','src')] + \
            ['starutil', 'starmem', 'cnf', 'ems', 'mers', 'chr',\
             'one'] + \
            [numpy.get_include()]

        from starlink import Ast
        ndfex_includedirs = hdsex_includedirs + ['prm', 'ast', 'ary', 'ast_missingincludes', Ast.get_include()]

        define_macros = get_starlink_macros(osx=osx)

        # Now build all.

        # This is the directory where the extra library's built here
        # have to be copied to, relative to the final build. This must
        # be called '.libs' if you want to use this with
        # cibuildwheel/auditwheel.
        extra_lib_dir = '.libs'

        # Get the compilers.
        compiler = ccompiler.new_compiler(dry_run=FAKEBUILDING, verbose=0)
        compiler2 = ccompiler.new_compiler(verbose=1)

        # Ensure we have any distutils options set.
        customize_compiler(compiler)
        customize_compiler(compiler2)


        # Now go through each extension, build the shared libraries we
        # need and ensure they are copied to the build directory. We
        # will use rpath and $ORIGIN to ensure everything is portable
        # as it will be moved around during the build process by pip
        # etc.
        for ext in self.extensions:
            linked_libraries = []
            if ext.name=='starlink.hds':


                hds_deps = compiler.compile(sources=hds_source_dep, output_dir=OUTPUTDIR,
                                         macros=define_macros, include_dirs=HDS_DEP_INCLUDES,
                                         depends=hds_source_dep,
                                        )

                hds_deps_libname = compiler2.library_filename('pyhdsdeps',lib_type=libtype)

                # Build this into a library

                print('Linking HDSDEPS\n\n\n')
                extra_preargs = None
                if osx:
                    extra_preargs = ['-v', '-Wl,-v',install_name_pattern.format(hds_deps_libname)]

                compiler2.link('shared', hds_deps, hds_deps_libname, output_dir=OUTPUTDIR, extra_preargs=extra_preargs, extra_postargs=None)
                linked_libraries += [os.path.join(OUTPUTDIR, hds_deps_libname)]

                # Now build hds-v4 and hds-v5: have to do this separately.
                hdsv4_libname = compiler.library_filename('pyhdsv4', lib_type=libtype)
                hdsv4objs = compiler.compile(sources = get_source('hds-v4'), output_dir=OUTPUTDIR,
                                                   macros=define_macros,
                                                   include_dirs=('hds-v4_missingincludes',) + HDS_DEP_INCLUDES,
                                                   depends=get_source('hds-v4'))
                extra_preargs = None
                if osx:
                    extra_preargs = [install_name_pattern.format(hdsv4_libname)]
                print('Linking HDSV4\n\n\n')
                compiler2.link('shared', hdsv4objs, hdsv4_libname, output_dir=OUTPUTDIR, extra_preargs=extra_preargs, target_lang='c')
                linked_libraries += [os.path.join(OUTPUTDIR, hdsv4_libname)]
                print('CREATING HDF5 LIBRARY\n\n\n')
                # Create the HDF5 library
                hdf5_libpath = os.path.join('hdf5', 'src', '.libs')
                hdf5_objects = glob.glob(os.path.join(hdf5_libpath, '*.o'))
                hdf5_libname = compiler.library_filename('pystarhdf5', lib_type=libtype)

                extra_preargs = None
                if osx:
                    extra_preargs = [install_name_pattern.format(hdf5_libname)]
                compiler2.link('shared', hdf5_objects, hdf5_libname,
                                   output_dir=OUTPUTDIR,
                                   library_dirs=[OUTPUTDIR],
                                   runtime_library_dirs=rdirs, extra_postargs=ldirs,
                                   extra_preargs=extra_preargs)
                linked_libraries += [os.path.join(OUTPUTDIR, hdf5_libname)]

                hdsv5_libname = compiler2.library_filename('pyhdsv5', lib_type=libtype)
                hdsv5objs = compiler.compile(sources = get_source('hds-v5'), output_dir=OUTPUTDIR,
                                                   macros=define_macros,
                                                   include_dirs=('hds-v5_missingincludes',) + HDS_DEP_INCLUDES,
                                                   depends=get_source('hds-v5'))
                extra_preargs = None
                if osx:
                    extra_preargs = [install_name_pattern.format(hdsv5_libname)]

                compiler2.link('shared', hdsv5objs, hdsv5_libname,
                                   output_dir=OUTPUTDIR,
                                   libraries=['pystarhdf5'],
                                   library_dirs=[OUTPUTDIR],
                                   runtime_library_dirs=rdirs,
                                   extra_postargs=ldirs,
                                   extra_preargs=extra_preargs)
                linked_libraries += [os.path.join(OUTPUTDIR, hdsv5_libname)]


                hds_libname = compiler2.library_filename('pyhds',lib_type=libtype)
                hdsobjs = compiler.compile(sources = get_source('hds'), output_dir=OUTPUTDIR,
                                            macros=define_macros, include_dirs=hdsex_includedirs,
                                            depends=get_source('hds'))
                extra_preargs = None
                if osx:
                    extra_preargs = [install_name_pattern.format(hds_libname)]

                compiler2.link('shared', hdsobjs, hds_libname,
                                   output_dir=OUTPUTDIR,
                                   libraries=['pyhdsdeps','pyhdsv5', 'pyhdsv4'],
                                   library_dirs=[OUTPUTDIR],
                                   runtime_library_dirs=rdirs, extra_postargs=ldirs,
                                   extra_preargs=extra_preargs)
                linked_libraries += [os.path.join(OUTPUTDIR, hds_libname)]

                ext.libraries += ['pyhds']
                ext.library_dirs += [OUTPUTDIR]
                if ext_runtime_library_dirs:
                    ext.runtime_library_dirs += [ext_runtime_library_dirs.format(extra_lib_dir)]
                if ext_extra_link_args:
                    ext.extra_link_args += [ext_extra_link_args.format(extra_lib_dir)]

            if ext.name=='starlink.ndf':
                ndf_libname = compiler2.library_filename('pyndf', lib_type=libtype)
                ndfobjs = compiler.compile(sources = get_source('ndf') + ndf_source_dep,
                                             include_dirs= ['ndf/', 'ndf_missingincludes/'] + ndfex_includedirs,
                                             macros=define_macros,
                                             depends=get_source('ndf'))

                extra_preargs = None
                if osx:
                    extra_preargs = [install_name_pattern.format(ndf_libname)]
                compiler2.link('shared', ndfobjs, ndf_libname,
                                   output_dir=OUTPUTDIR,
                                   libraries=['pyhdsdeps','pyhdsv5', 'pyhdsv4', 'pyhds'],
                                   library_dirs=[OUTPUTDIR],
                                   runtime_library_dirs=rdirs,
                                   extra_postargs=ldirs,
                                   extra_preargs=extra_preargs)
                linked_libraries += [os.path.join(OUTPUTDIR, ndf_libname)]
                ext.libraries += ['pyndf']
                ext.library_dirs += [OUTPUTDIR]
                if ext_runtime_library_dirs:
                    ext.runtime_library_dirs += [ext_runtime_library_dirs.format(extra_lib_dir)]
                if ext_extra_link_args:
                    ext.extra_link_args += [ext_extra_link_args.format(extra_lib_dir)]

            # Copy over the libraries to the build directory manually, and add to package data.
            if not os.path.isdir(os.path.join(self.build_lib, 'starlink', extra_lib_dir)):
                os.mkdir(os.path.join(self.build_lib, 'starlink', extra_lib_dir))

            for lib in linked_libraries:
                shutil.copy(lib, os.path.join(self.build_lib, 'starlink', extra_lib_dir))
                output_lib = os.path.join('starlink', extra_lib_dir, os.path.split(lib)[1])
                self.distribution.package_data.get('starlink', list()).extend(output_lib)

        # Run the standard build_ext process.
        build_ext.run(self)



# Get the Starlink specific defines.
defines = get_starlink_macros()


# Get the lists of source files for the NDF extra dependencies.
ndf_source_dep = []
for name_ in ['prm', 'ast', 'ary']:
    ndf_source_dep += get_source(name_)



hdsex_includedirs = ['include/', 'hds/', 'missingincludes/',
                     'hds_missingincludes/', 'hdf5/src/', 'hdf5/hl/src'] + \
    ['starutil', 'starmem/', 'cnf', 'ems', 'mers', 'chr',\
     'one'] + \
    [numpy.get_include()]



ndfex_includedirs = hdsex_includedirs + ['prm', 'ast', 'ary', 'ast_missingincludes/', Ast.get_include()]

# Can't build NDF without Ast!


# Define the two extensions.

hdsExtension = Extension('starlink.hds',
                         sources = ['starlink/hds.pyx'],
                         include_dirs = hdsex_includedirs,
                         define_macros = defines,
                         libraries = ['z'],
)


ndfExtension = Extension('starlink.ndf',
                         sources = ['starlink/ndf.pyx'],
                         include_dirs = ['ndf/', 'ndf_missingincludes/'] + ndfex_includedirs,
                         define_macros = defines,
                         libraries = ['z'],
)



with open('README.rst') as file:
    long_description = file.read()

setup(name='starlink-pyndf',
      version='0.3',
      long_description=long_description,
      packages=['starlink', 'starlink.ndfpack'],
      cmdclass={'build_ext': custom_star_build},
      ext_modules=cythonize([hdsExtension, ndfExtension],
                            language_level=2,
                            gdb_debug=True,
                            include_path=[
                                'starlink/', Ast.get_include()],
                            compiler_directives={'embedsignature':True}),

      package_data={'starlink.hds':['starlink/hds.pxd'],'starlink.ndf':['starlink/ndf.pxd']},

      test_suite='test',
      namespace_packages=['starlink'],

      # metadata
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

      setup_requires = ['numpy', 'starlink-pyast'],
      install_requires = ['numpy','starlink-pyast'],
      test_requires = ['pathlib'],

)
