from __future__ import print_function

from distutils import ccompiler, sysconfig
from setuptools import setup
from setuptools import Extension
from distutils.command.build_ext import build_ext
import glob
import sys
import time
import os
import numpy as np
import ctypes


import subprocess
"""
Setup script for the hds python extension.
"""

# The hds library requires: starmem, ems, hds and sae to build. Then
#  hds-v4 and hds-v5.  These must all be built before hds is. hds-v5
#  requires hdf5_path -- can this use installed version??

hdsv4_path = 'hds-v4-5.2-1/'
hdsv5_path = 'hds-v5-1.0-1/'
hds_path = 'hds-6.0-1/'

ems_path = 'ems-2.4-0/'
starmem_path = 'starmem-0.2-1/'
sae_path = 'sae-1.1/'
cnf_path = 'cnf-5.1-0/'
mers_path = 'mers-2.2-0/'
starutil_path = 'starutil-0.1-1/'
hdf5_path = 'star-thirdparty-hdfgroup-1.0'
one_path = 'one-1.5-1'



# Get the source files. Odds and ends (one) repo:
one_sources = glob.glob(os.path.join(one_path, '*.c'))
one_sources.remove(os.path.join(one_path, 'cone_test.c'))
one_sources.remove(os.path.join(one_path, 'one_wordexp_noglob.c'))
one_sources.remove(os.path.join(one_path, 'one_wordexp_file.c'))

# starmem: these sources are taking from the Makefile.am after doing a
# make dist in starmem.

starmem_PUBLIC_C_FILES = [
    'starMalloc.c',
    'starMallocAtomic.c',
    'starMemInitPrivate.c',
    'starFree.c',
    'starFreeForce.c',
    'starRealloc.c',
    'starCalloc.c',
    'starMemIsInitialised.c'
]

starmem_PRIVATE_C_FILES = ['mem1_globals.c', 'dlmalloc.c']
starmem_sources = [os.path.join(starmem_path, i)
                  for i in
                  starmem_PRIVATE_C_FILES + starmem_PUBLIC_C_FILES]

# ems
ems_C_ROUTINES=[
    'emsAnnul.c', 'emsBegin.c', 'emsEload.c', 'emsEnd.c', 'emsErrno.c',
    'emsExpnd.c', 'emsFacer.c', 'emsLevel.c', 'emsMark.c', 'emsMload.c',
    'emsRenew.c', 'emsRep.c', 'emsRlse.c', 'emsSetc.c', 'emsSetd.c',
    'emsSeti.c', 'emsSetk.c', 'emsSetl.c', 'emsSetnc.c', 'emsSetp.c',
    'emsSetr.c', 'emsSetu.c', 'emsStat.c', 'emsSyser.c', 'emsTune.c',
    'emsGtune.c', 'emsStune.c',
    'emsSet.c', 'emsSetv.c',
    'ems1Eblk.c', 'ems1Emark.c', 'ems1Erlse.c', 'ems1Estor.c', 'ems1Estor1.c',
    'ems1Fcerr.c', 'ems1Flush.c', 'ems1Form.c', 'ems1Fthreaddata.c', 'ems1Gesc.c',
    'ems1Gmsgtab.c', 'ems1Gmsgtab2.c', 'ems1Gnam.c', 'ems1Gtok.c', 'ems1Gtoktab.c',
    'ems1Gthreadbuf.c', 'ems1Iepnd.c', 'ems1Imsgtab.c', 'ems1Ithreaddata.c',
    'ems1Itoktab.c', 'ems1Kerr.c', 'ems1Ktok.c', 'ems1Mpop.c', 'ems1Mpush.c',
    'ems1Prerr.c', 'ems1Putc.c', 'ems1Rform.c', 'ems1Serr.c', 'ems1Starf.c',
    'ems1Stok.c', 'ems1Tblk.c', 'ems1Rep.c', 'emsRepv.c', 'emsRepf.c']

ems_sources = [os.path.join(ems_path, i)
                  for i in
                  ems_C_ROUTINES]

#starutil
starutil_PUBLIC_C_FILES=['star_strlcat.c', 'star_strlcpy.c', 'star_strappend.c',
	'star_strellcpy.c']
starutil_sources = [os.path.join(starutil_path, i) for i in starutil_PUBLIC_C_FILES]



#cnf
cnf_C_ROUTINES = ['cnfCopyf.c', 'cnfCreat.c', 'cnfCref.c', 'cnfCrefa.c', 'cnfCreib.c',
    'cnfCreim.c', 'cnfCrela.c',
    'cnfExpch.c', 'cnfExpla.c', 'cnfExpn.c', 'cnfExprt.c', 'cnfExprta.c', 'cnfExprtap.c',
    'cnfFreef.c',
    'cnfImpb.c', 'cnfImpbn.c', 'cnfImpch.c', 'cnfImpla.c', 'cnfImpn.c', 'cnfImprt.c',
    'cnfImprta.c', 'cnfImprtap.c', 'cnfLenc.c', 'cnfLenf.c',
    'cnfMem.c', 'cnfLock.c']
cnf_sources = [os.path.join(cnf_path, i) for i in cnf_C_ROUTINES]

#mers
mers_C_INTERFACE_ROUTINES = [
'errAnnul.c',
'errBegin.c',
'errEnd.c',
'errFacer.c',
'errFlbel.c',
'errFlush.c',
'errLevel.c',
'errLoad.c',
'errMark.c',
'errOut.c',
'errRep.c',
'errRepf.c',
'errRlse.c',
'errStat.c',
'errSyser.c',
'errTune.c',
'msgBell.c',
'msgBlank.c',
'msgBlankif.c',
'msgFlevok.c',
'msgFlusherr.c',
'msgFmt.c',
'msgIfgetenv.c',
'msgIflev.c',
'msgIfset.c',
'msgLoad.c',
'msgOut.c',
'msgOutf.c',
'msgOutif.c',
'msgOutiff.c',
'msgOutifv.c',
'msgRenew.c',
'msgSetc.c',
'msgSetd.c',
'msgSeti.c',
'msgSetk.c',
'msgSetl.c',
'msgSetr.c',
'msgTune.c']

mers_C_ROUTINES_STAND = [
'err1Prerr_stand.c',
'msg1Form_stand.c',
'msg1Prtln_stand.c',
'msgSync_stand.c',
]
mers_C_ROUTINES = [
'err1Bell.c',
'err1Flush.c',
'err1Print.c',
'err1Rep.c',
'mers1Blk.c',
'mers1Getenv.c',
'msg1Ifget.c',
'msg1Ktok.c',
'msg1Levstr.c',
'msg1Outif.c',
'msg1Print.c']

mers_C_ROUTINES_ADAM = [
'err1Prerr_adam.c',
'msg1Form_adam.c',
'msg1Genv_adam.c',
'msg1Gkey_adam.c',
'msg1Gref_adam.c',
'msg1Prtln_adam.c']

mers_C_INTERFACE_ADAM = [
'errClear_adam.c',
'errStart_adam.c',
'errStop_adam.c',
'msgIfget_adam.c',
'msgSync_adam.c']

mers_sources = [os.path.join(mers_path, i) for i in
                mers_C_INTERFACE_ROUTINES +
                mers_C_ROUTINES_STAND + mers_C_ROUTINES]

hds_C_ROUTINES = glob.glob(os.path.join(hds_path, '*.c'))
hds_sources = hds_C_ROUTINES

hds_sources.remove(os.path.join(hds_path, 'hdsTest.c'))
hds_sources.remove(os.path.join(hds_path, 'make-hds-types.c'))
hds_sources.remove(os.path.join(hds_path, 'fortran_interface.c'))
hds_sources.remove(os.path.join(hds_path, 'hdsdim.c'))
hds_sources.remove(os.path.join(hds_path, 'hdsDimtoc.c'))
hds_sources.remove(os.path.join(hds_path, 'hdsFind.c'))
hds_sources.remove(os.path.join(hds_path, 'hdsSplit.c'))
hds_sources.remove(os.path.join(hds_path, 'hds_split.c'))
hds_sources.remove(os.path.join(hds_path, 'hds_run.c'))


# We now need to replicate the configure action. I'm doing this
# manually witht he compiler,but I'm very sure there is a better and
# more automated way to do this from the configure files.

# Test the compiler: find all the #undef commands from the config.h.in files
define_macros = []
compiler = ccompiler.new_compiler()
# The various sizeOF definitions.

#undef SIZEOF_INT
define_macros.append(('SIZEOF_INT', ctypes.sizeof(ctypes.c_int)))

#undef SIZEOF_LONG
define_macros.append(('SIZEOF_LONG', ctypes.sizeof(ctypes.c_long)))

#undef SIZEOF_LONG_DOUBLE
define_macros.append(('SIZEOF_LONG_DOUBLE', ctypes.sizeof(ctypes.c_longdouble)))

#undef SIZEOF_LONG_LONG
define_macros.append(('SIZEOF_LONG_LONG', ctypes.sizeof(ctypes.c_longlong)))

#undef SIZEOF_OFF_T
# don't know how to do this one.

#undef SIZEOF_SIZE_T
define_macros.append(('SIZEOF_SIZE_T', ctypes.sizeof(ctypes.c_size_t)))

#undef SIZEOF_TRAILARG
#I THINK THIS is only needed in the fortran interfaces?

#undef SIZEOF_UINT32_T
define_macros.append(('SIZEOF_UINT32_T', ctypes.sizeof(ctypes.c_uint32)))
#undef SIZEOF_VOIDP
define_macros.append(('SIZEOF_VOIDP', ctypes.sizeof(ctypes.c_voidp)))

#undef AC_APPLE_UNIVERSAL_BUILD
#undef FC_MAIN
#only needed in fortran?


#undef HAVE_ATEXIT
if compiler.has_function('atexit'):
    define_macros.append(('HAVE_ATEXIT', '1'))

#undef HAVE_BCOPY
#bcopy is on both test OSX and Linux systems, but has_function on
# linux raises an 'error: too few arguments to function
# 'bcopy'',. Works fine on clang. However, bcopy only used if memmove
# doesn't exist, so maybe don't worry about it?
if compiler.has_function('bcopy'):
    define_macros.append(('HAVE_BCOPY', '1'))

#undef HAVE_CRT_EXTERNS_H
# Not clear how to do. It looks like OSX should ahve these, but I
# don't know if that is universal. Defaulting to not setting for now?

#undef HAVE_DECL_CONFSTR
if compiler.has_function('confstr'):
    define_macros.append(('HAVE_DECL_CONFSTR', '1'))


#These systems aren't quite right: not sure how to check for declaration?
#undef HAVE_DECL_FSYNC
if compiler.has_function('fsync'):
    define_macros.append(('HAVE_DECL_FSYNC', '1'))

#undef HAVE_DECL_GETWD
#This one I may not want to declare? Looks like most systems think it isn't...

#undef HAVE_DECL_STRERROR_R
# Don't know what to do with this.


#undef HAVE_DLFCN_H
#Only used in configuring scripts???

#undef HAVE_FCNTL
if compiler.has_function('fcntl'):
    define_macros.append(('HAVE_FCNTL', '1'))

#undef HAVE_FC_MAIN
# Only needed in fortran?

#undef HAVE_FORK
#Not used.

#undef HAVE_FSEEKO
if compiler.has_function('fseeko'):
    define_macros.append(('HAVE_FSEEKO', '1'))

#undef HAVE_FSYNC
if compiler.has_function('fsync'):
    define_macros.append(('HAVE_FSYNC', '1'))

#undef HAVE_GC_H
# Not clear how to do these ones, but currently looks like both our
# linux and our OSX build don't set them, so assume its okay to skip
# for now.


#undef HAVE_GETCWD
if compiler.has_function('getcwd'):
    define_macros.append(('HAVE_GETCWD', '1'))

#undef HAVE_GETPAGESIZE
if compiler.has_function('getpagesize'):
    define_macros.append(('HAVE_GETPAGESIZE', '1'))

#undef HAVE_GETWD
if compiler.has_function('getwd'):
    define_macros.append(('HAVE_GETWD', '1'))

#undef HAVE_INT
#Not used, but in configure.

#undef HAVE_INT32_T
#Not used, but in configure.

#undef HAVE_INT64_T
#Only used to make-hds-types.c, which we don't do here (done in make dist of hds)

#undef HAVE_INTTYPES_H
if sysconfig.get_config_var('HAVE_INTTYPES_H')==1:
    define_macros.append(('HAVE_INTTYPES_H', '1'))

#undef HAVE_LIBGC
# Not clear how to do these ones, but currently looks like both our
# linux and our OSX build don't set them, so assume its okay to skip
# for now.

#undef HAVE_LIBPTHREAD
#Set in configure scripts but not used?

#undef HAVE_LONG
#Set in configure scripts but not used?

#undef HAVE_LONG_DOUBLE
if sysconfig.get_config_var('HAVE_LONG_DOUBLE')==1:
    define_macros.append(('HAVE_LONG_DOUBLE', '1'))

#undef HAVE_LONG_LONG
#Set in configure scripts but not used?

#undef HAVE_LONG_LONG_INT
#Set in configure scripts but not used?

#undef HAVE_LONG_LONG_INT
#Set in configure scripts but not used?

#undef HAVE_MEMMOVE
define_macros.append(('HAVE_MEMMOVE', '1'))
# Cannot get this to work test... fix would be to compile a short bit
# of code that uses it correctly?

#undef HAVE_MEMORY_H
#Set in configure scripts but not used?

#undef HAVE_MMAP
if compiler.has_function('mmap'):
    define_macros.append(('HAVE_MMAP', '1'))


#undef HAVE_OFF_T
#Set in configure scripts but not used?

#undef HAVE_ON_EXIT
if compiler.has_function('on_exit'):
    define_macros.append(('HAVE_ON_EXIT', '1'))

#undef HAVE_PERCENTLOC
#Only needed for fortran

#undef HAVE_SIGNED_CHAR
# Not sure how to test for this. Try assuming its true? (note that
# lots of the code uses unsignedchar without checking...)
define_macros.append(('HAVE_SIGNED_CHAR', '1'))


#undef HAVE_STDDEF_H

#Define to 1 if you have the <stddef.h> header file.
# Lots of things just include it without paying attention tot his
# variable, so just define it as true.
define_macros.append(('HAVE_STDDEF_H', '1'))


#undef HAVE_STDINT_H
#Only used in make-hds-types.c

#undef HAVE_STDLIB_H
#Set in configure scripts but not used?

#undef HAVE_STRERROR_R
if compiler.has_function('strerror_r'):
    define_macros.append(('HAVE_STRERROR_R', '1'))

#undef HAVE_STRINGS_H
#Set in configure scripts but not used?

#undef HAVE_STRING_H
#Set in configure scripts but not used?

#undef HAVE_STRTOK_R
if compiler.has_function('strtok_r'):
    define_macros.append(('HAVE_STRTOK_R', '1'))

#undef HAVE_SYS_PARAM_H
#Set in configure scripts but not used?

#undef HAVE_SYS_STAT_H
#Set in configure scripts but not used?

#undef HAVE_SYS_TYPES_H
#Set in configure scripts but not used?

#undef HAVE_SYS_WAIT_H
if sysconfig.get_config_var('HAVE_SYS_WAIT_H')==1:
    define_macros.append(('HAVE_SYS_WAIT_H', '1'))

#undef HAVE_TIME_H
#Only used in make-hds-types.c

#undef HAVE_UINT32_T
#Set in configure scripts but not used?

#undef HAVE_UINT64_T
#Only used in make-hds-types.c

#undef HAVE_UNISTD_H
if sysconfig.get_config_var('HAVE_UNISTD_H')==1:
    define_macros.append(('HAVE_UNISTD_H', '1'))

#undef HAVE_VFORK
#Set in configure scripts but not used?

#undef HAVE_VFORK_H
#Set in configure scripts but not used?

#undef HAVE_WORKING_FORK
#Set in configure scripts but not used?

#undef HAVE_WORKING_VFORK
#Not sure how to set this: only used in hds1.h

#undef HAVE__NSGETENVIRON
# Not clear how to do. It looks like OSX should ahve these, but I
# don't know if that is universal. Defaulting to not setting for now?
# Used in rec1_shell.c

#undef LT_OBJDIR
#Set in configure scripts but not used?

#undef STAR_INITIALISE_FORTRAN
#Only neededin fortran

#undef STDC_HEADERS
# Set in configure scripts but not used?

#undef STRERROR_R_CHAR_P
# Set in configure scripts but not used?

#undef TRAIL_TYPE
#Used in fortran, but probably needed to build -- set to int, which is what configure sets it to on linux.
define_macros.append(('TRAIL_TYPE', 'int'))

#undef USE_PTHREADS
#Build with POSIX threads support? Probably want this, but not sure best way to do check...

#undef _FILE_OFFSET_BITS
# Set in configure scripts but not used? Not set on linux anyway

#undef _LARGEFILE_SOURCE
# Set in configure scripts but not used? Not set on linux anyway

#undef _LARGE_FILES
# Set in configure scripts but not used? Not set on linux anyway

#undef _POSIX_C_SOURCE
#Doesn't look like it can change. Breaks OSX build so skip.
#define_macros.append(('_POSIX_C_SOURCE', '200112L'))

#undef _REENTRANT
# should be defined probably if use_pthreads is defined?

#undef const
#Defined to empty if 'const' doesn't conform to ANSI C? Probably skip for now.

#undef pid_t
#Needs to be defined to int if it is not already defined -- leave for now?

#undef vfork
#I think this can be skipped? Nothing actually calls vfork without checking?




#Now set up the include directories?
include_dirs = []
include_dirs.append(np.get_include())
include_dirs.append(os.path.join('.', 'includefiles'))
include_dirs.append(os.path.join(hdf5_path,'hdf5/hdf5/src/'))
include_dirs.append(os.path.join(hdf5_path,'hdf5/hdf5/hl/src/'))


include_dirs.append(os.path.join('.', 'starlink', 'hds'))

include_dirs += [os.path.join('.', i) for i in [starmem_path, ems_path,
                                                one_path, sae_path,
                                                cnf_path, hds_path]]

#Now set up all the source files, starting with the main modules and
#then all the .c files needed to build the libraries
sources = [os.path.join('starlink', 'hds', 'hds.c')]

sources += starmem_sources
sources += starutil_sources
sources += one_sources
sources += cnf_sources
sources += ems_sources
sources += mers_sources
sources += hds_sources

define_macros.append(('HDS_INTERNAL_INCLUDES', '1'))
#define_macros.append(('SAI__OK', '0'))
define_macros.append(('ERR__SZMSG', '200'))
define_macros.append(('ERR__SZPAR', '15'))
define_macros.append(('_GNU_SOURCE', 1))




# Now set up the Extension.
hds = Extension('starlink.hds',
                define_macros        = define_macros,
                include_dirs         = include_dirs,
                sources              = sources,
                libraries = ['z'],
                )

# Set up the custom build_ext options, to call ./configure and make
# for hdf5, hdsv4 and hdsv5.
def configuremake(path, cppflags=None, ldflags=None,
                  maketargets=None, environ_override=None):
    basedir = os.getcwd()
    os.chdir(path)
    env = os.environ


    # We ahve to touch the files to ensure they have the write timestamps.
    fnames = ['ltmain.sh',
              'missing',
              'install-sh',
              'config.sub',
              'config.guess',
              'compile',
              'configure.ac', 'Makefile.am', 'aclocal.m4', 'configure', 'Makefile.in']
    for fn in fnames:
        if os.path.isfile(fn):
            os.utime(fn, None)
            time.sleep(1)

    if cppflags:
        env['CPPFLAGS']=cppflags

    if ldflags:
        env['LDFLAGS']=ldflags
    subprocess.check_call('./configure', env=env)

    if environ_override:
        for key in environ_override:
            env[key] = environ_override[key]
    if not maketargets:
        print('Running make.')
        if environ_override:
            subprocess.check_call(['make', '-e'], env=env)
        else:
            subprocess.check_call('make', env=env)
    else:
        for target in maketargets:
            print('Running make {}'.format(target))
            if environ_override:
                subprocess.check_call(['make', '-e', target], env=env)
            else:
                subprocess.check_call(['make', target], env=env)
    os.chdir(basedir)


class custom_build(build_ext):
    def run(self):

        # Before we can build the extension, we have to run
        # ./configure and make for hdf5, hdsv4 and hdsv5.
        print('\n\nBuilding HDF5')
        configuremake(os.path.join(hdf5_path, 'hdf5'))

        incfiles = os.path.join(os.pardir, 'includefiles')
        print('\n\nBuilding HDSV4')
        configuremake(hdsv4_path, cppflags='-I{}'.format(incfiles),
                      maketargets=['hds_types.h', 'libhds_v4.la'])

        print('\n\nBuilding HDSV5')

        hdf5loc = os.path.abspath(os.path.join(hdf5_path,
                'hdf5', 'hdf5', 'src'))
        hdf5hlloc = os.path.abspath(os.path.join(hdf5_path,
                'hdf5', 'hdf5', 'hl', 'src'))

        hdf5lib_loc = (os.path.join(hdf5loc, '.libs'))
        hdf5lib = os.path.join(hdf5lib_loc, 'libhdf5.so')

        environ_override = {'libhds_v5_la_LIBADD': ''}
        configuremake(hdsv5_path, cppflags='-I{} -I{} -I{}'.format(incfiles,
                                                                   hdf5loc,
                                                                   hdf5hlloc),
                      environ_override = environ_override,
                      maketargets=['hds_types.h', 'libhds_v5.la'])

        # We now need to add all of the appropriate .o files to the
        # extra_objects option for this extension.
        extras = glob.glob(os.path.join(hdf5_path,
                                'hdf5','hdf5','src', '.libs', '*.o'))
        extras += glob.glob(os.path.join(hdsv4_path, '*.o'))
        extras += glob.glob(os.path.join(hdsv5_path, '*.o'))
        extras.remove(os.path.join(hdsv4_path, 'make-hds-types.o'))
        extras.remove(os.path.join(hdsv5_path, 'make-hds-types.o'))

        self.extensions[0].extra_objects = extras

        print('\n\n\n Building main hds extension')
        build_ext.run(self)


with open('README.rst') as file:
    long_description = file.read()




setup(name='starlink-pyhds',
      version='0.2.rc1',
      description='Python interface to the Starlink HDS library',
      long_description=long_description,
      packages=['starlink'],
      cmdclass={'build_ext': custom_build},
      ext_modules=[hds],
      test_suite='test',

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
      install_requires = [
          'numpy',
          ],
      )
