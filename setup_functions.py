import ctypes
from distutils import ccompiler, sysconfig
import glob
import os
import shutil


def get_starlink_macros(osx=False):
    # Need to define: HAVE_INT64_T && HAVE_UINT64_T
    define_macros = []
    compiler = ccompiler.new_compiler()

    define_macros.append(('NDF_I8', 1))
    define_macros.append(('HAVE_AST', 1))

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


    #undef HAVE_INTTYPES_H
    if sysconfig.get_config_var('HAVE_INTTYPES_H')==1:
        define_macros.append(('HAVE_INTTYPES_H', '1'))



    #undef HAVE_LONG_DOUBLE
    if sysconfig.get_config_var('HAVE_LONG_DOUBLE')==1:
        define_macros.append(('HAVE_LONG_DOUBLE', '1'))

    #

    #undef HAVE_MEMMOVE
    define_macros.append(('HAVE_MEMMOVE', '1'))
    # Cannot get this to work test... fix would be to compile a short bit
    # of code that uses it correctly?



    #undef HAVE_MMAP
    if compiler.has_function('mmap'):
        define_macros.append(('HAVE_MMAP', '1'))


    #undef HAVE_ON_EXIT
    if compiler.has_function('on_exit'):
        define_macros.append(('HAVE_ON_EXIT', '1'))



    #undef HAVE_SIGNED_CHAR
    # Not sure how to test for this. Try assuming its true? (note that
    # lots of the code uses unsignedchar without checking...)
    define_macros.append(('HAVE_SIGNED_CHAR', '1'))



    #Define to 1 if you have the <stddef.h> header file.
    # Lots of things just include it without paying attention tot his
    # variable, so just define it as true.
    define_macros.append(('HAVE_STDDEF_H', '1'))

    #For  OSX, we need the following macros.
    if osx:
        define_macros.append(('HAVE_STRLCAT', '1'))
        define_macros.append(('HAVE_STRLCPY', '1'))

    #undef HAVE_STRERROR_R
    if compiler.has_function('strerror_r'):
        define_macros.append(('HAVE_STRERROR_R', '1'))

    #undef HAVE_STRTOK_R
    if compiler.has_function('strtok_r'):
        define_macros.append(('HAVE_STRTOK_R', '1'))

    #undef HAVE_SYS_WAIT_H
    if sysconfig.get_config_var('HAVE_SYS_WAIT_H')==1:
        define_macros.append(('HAVE_SYS_WAIT_H', '1'))

    #undef HAVE_UNISTD_H
    if sysconfig.get_config_var('HAVE_UNISTD_H')==1:
        define_macros.append(('HAVE_UNISTD_H', '1'))
    #undef TRAIL_TYPE
    #Used in fortran, but probably needed to build -- set to int, which is what configure sets it to on linux.
    define_macros.append(('TRAIL_TYPE', 'int'))

    # various others.
    define_macros.append(('HAVE_INT64_T', 1))
    define_macros.append(('HAVE_UINT64_T', 1))
    define_macros.append(('HAVE_DECL_ISFINITE', 1))
    define_macros.append(('HAVE_DECL_ISNAN', 1))
    define_macros.append(('HDS_INTERNAL_INCLUDES', '1'))
    define_macros.append(('ERR__SZMSG', '200'))
    define_macros.append(('ERR__SZPAR', '15'))
    define_macros.append(('_GNU_SOURCE', 1))
    #define_macros.append(('HAVE_CUSERID', 1))
    define_macros.append(('HAVE_DECL_CUSERID_GETUSERNAME', 0))
    #define_macros.append(('HAVE_DECL_CUSERID',0))



    if compiler.has_function('cuserid'):
        define_macros.append(('HAVE_CUSERID', '1'))
    if compiler.has_function('getlogin'):
        define_macros.append(('HAVE_GETLOGIN', '1'))
    if compiler.has_function('getpwuid'):
        define_macros.append(('HAVE_GETPWUID', '1'))
    if compiler.has_function('geteuid'):
        define_macros.append(('HAVE_GETEUID', '1'))

    #define_macros.append(('DEBUG_HDS', '1'))
    return define_macros



def get_source(name_):
    """
    Get the source lists for libraries.

    This should be black lists of files to exclude, so that if more
    code is added that can't build the build will break and tell us we
    need to deal with it, rather than have us leave things out.

    (This decision is predicated on the fact that these libraries
    aren't changing particularly often.)
    """
    if name_ == 'ndf':
        with open('ndf_csource.lis', 'r') as f:
            source = ['ndf/' + i.strip() for i in f.readlines()]

    elif name_ == 'ast':
        cminpack_source = ['enorm.c', 'lmder.c', 'qrfac.c', 'dpmpar.c', 'lmder1.c',
                           'lmpar.c', 'qrsolv.c' ]
        cminpack_source = ['ast/cminpack/' + i for i in cminpack_source]
        ast_exclude_files = ['ast/ast_test.c', 'ast/c2f77.c', 'ast/err_drama.c', 'ast/grf_null.c', 'ast/grf_pgplot.c',  'ast/grf3d_pgplot.c', 'ast/stcschan-demo1.c', 'ast/stcschan-demo2.c', 'ast/stcschan-demo3.c', 'ast/stcschan-demo4.c', 'ast/stcschan-demo5.c', 'ast/templateclass.c', 'ast/err_null.c', 'ast/huge.c', 'ast/astbad.c',]
        erfa_exclude_files = ['ast/erfa/t_erfa_c.c',]
        ast_fortran_files = [
            'ast/fbox.c',
            'ast/fchannel.c',
            'ast/fchebymap.c',
        'ast/fcircle.c',
            'ast/fcmpframe.c',
            'ast/fcmpmap.c',
            'ast/fcmpregion.c',
            'ast/fdsbspecframe.c',
            'ast/fdssmap.c',
            'ast/fellipse.c',
            'ast/ferror.c',
            'ast/ffitschan.c',
            'ast/ffitstable.c',
            'ast/ffluxframe.c',
            'ast/fframe.c',
            'ast/fframeset.c',
            'ast/fgrismmap.c',
            'ast/finterval.c',
            'ast/fintramap.c',
            'ast/fkeymap.c',
            'ast/flutmap.c',
            'ast/fmapping.c',
            'ast/fmathmap.c',
            'ast/fmatrixmap.c',
            'ast/fmoc.c',
            'ast/fmocchan.c',
            'ast/fnormmap.c',
            'ast/fnullregion.c',
            'ast/fobject.c',
            'ast/fpcdmap.c',
            'ast/fpermmap.c',
            'ast/fplot.c',
            'ast/fplot3d.c',
            'ast/fpointlist.c',
            'ast/fpolygon.c',
            'ast/fpolymap.c',
            'ast/fprism.c',
            'ast/fratemap.c',
            'ast/fregion.c',
            'ast/fselectormap.c',
            'ast/fshiftmap.c',
            'ast/fskyframe.c',
            'ast/fslamap.c',
            'ast/fspecfluxframe.c',
            'ast/fspecframe.c',
            'ast/fspecmap.c',
            'ast/fsphmap.c',
            'ast/fstc.c',
            'ast/fstccatalogentrylocation.c',
            'ast/fstcobsdatalocation.c',
            'ast/fstcresourceprofile.c',
            'ast/fstcschan.c',
            'ast/fstcsearchlocation.c',
            'ast/fswitchmap.c',
            'ast/ftable.c',
            'ast/ftemplateclass.c',
            'ast/ftimeframe.c',
            'ast/ftimemap.c',
            'ast/ftranmap.c',
            'ast/funitmap.c',
            'ast/funitnormmap.c',
            'ast/fwcsmap.c',
            'ast/fwinmap.c',
            'ast/fxmlchan.c',
            'ast/fzoommap.c',
        ]
        ast_exclude_files += ast_fortran_files

        ast_source = glob.glob('ast/*.c')
        erfa_source = glob.glob('ast/erfa/*.c')
        ast_source = [i for i in ast_source if i not in ast_exclude_files]
        erfa_source = [i for i in erfa_source if i not in erfa_exclude_files]

        source = ast_source + erfa_source + cminpack_source

    elif name_ == 'starutil':
        source = glob.glob('starutil/star_*.c')
    elif name_ == 'starmem':

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
        source = [os.path.join(name_, i)
                          for i in
                          starmem_PRIVATE_C_FILES + starmem_PUBLIC_C_FILES]

    elif name_ == 'one':

        source = [
        'one/one_snprintf.c',
        'one/one_strlcat.c',
        'one/one_strlcpy.c',
        'one/one_strtod.c',
        'one/one_wordexp_noglob_c.c',
        ]

    elif name_ == 'chr':
        source = glob.glob('chr/chr*.c')
        source = [i for i in source if 'test' not in i.lower()]

    elif name_ == 'prm':
        source = glob.glob('prm/*.c')
    elif name_ == 'cnf':
        source = glob.glob('cnf/cnf*.c')
        source = [i for i in source if 'test' not in i and 'cnfInitRTL' not in i]

    elif name_ == 'ems':
        source = glob.glob('ems/*.c')
        source = [i for i in source if '_' not in i]

    elif name_ == 'mers':
        mers_stand = ['mers/err1Prerr_stand.c', 'mers/msg1Form_stand.c',
                      'mers/msg1Prtln_stand.c', 'mers/msgSync_stand.c']
        source = glob.glob('mers/*.c')
        source = [i for i in source if 'adam' not in i]
        source = [i for i in source if ('_' not in i or i in mers_stand)]


    elif name_ == 'hds-v4':
        source = glob.glob('hds-v4/*.c')
        skip = ['hds-v4/make-hds-types.c','hds-v4/hdsTest.c', 'hds-v4/hds_machine.c', 'hds-v4/hds_test_prm.c']

        source = [i for i in source if i not in skip]

    elif name_ == 'hds-v5':
        source = glob.glob('hds-v5/*.c')
        skip = ['hds-v5/make-hds-types.c','hds-v5/hdsTest.c', 'hds-v5/hds_machine.c', 'hds-v5/fortran_interface.c', 'hds-v5/datExportFloc.c', 'hds-v5/datImportFloc.c']
        source = [i for i in source if 'test' not in i and i not in skip]

    elif name_ == 'hds':
        source = glob.glob('hds/*.c')
        source = [i for i in source if 'fortran' not in i.lower() and 'test' not in i.lower()]
        exclude = ['hds/hds_dat_par_f.c', 'hds/dat_par_f.c', 'hds/datLocked.c','hds/datLock.c', 'hds/datUnlock.c', 'hds/datNolock.c', 'hds/dat1emsSetHdsdim.c', 'hds/make-hds-types.c', 'hds/hds_run.c', 'hds/fortran_interface.c']
        source = [i for i in source if i not in exclude]

    elif name_ == 'ary':
        source = glob.glob('ary/ary*.c')
        exclude = ['ary/aryTest.c']
        source = [i for i in source if i not in exclude]

    else:
        raise StandardError('Unknown library name %s; source list not supported for this library', name_)


    return source



def setup_building():
    #  a) ensure we have an include directory
    if not os.path.isdir('include'):
        os.mkdir('include')
    if not os.path.isdir(os.path.join('include', 'star')):
        os.mkdir(os.path.join('include', 'star'))

    # b) cp cgen files into relevant directories.
    cgenfiles = glob.glob('ndf_cgenfiles/*')
    for i in cgenfiles:
        shutil.copy(i, 'ndf/')
    cgenfiles = glob.glob('ary_cgenfiles/*')
    for i in cgenfiles:
        shutil.copy(i, 'ary/')

    # c) Copy the public H files that are in our sourcecode into the include
    # directory.
    shutil.copy('ary/ary.h', 'include/')
    shutil.copy('ary/ary_types.h', 'include/')
    # Include the public ary_cgen.h that is already in our cgenfiles dir.
    shutil.copy('ary_cgenfiles/ary_cgen.h', 'include/')

    shutil.copy('ems/ems.h', 'include/')
    shutil.copy('ems/ems_par.h', 'include/')

    shutil.copy('one/one.h', 'include/')
    shutil.copy('one/one.h', 'include/star')

    shutil.copy('prm/prm_par.h', 'include/')
    shutil.copy('prm/prm.h', 'include/')
    shutil.copy('prm/prm_cgen.h', 'include/')

    shutil.copy('mers/mers.h', 'include/')
    shutil.copy('mers/merswrap.h', 'include/')
    shutil.copy('mers/msg_par.h', 'include/')
    shutil.copy('mers/err_par.h', 'include/')
    shutil.copy('mers/star/subpar.h', 'include/star/')

    shutil.copy('hds/dat_par.h', 'include/')
    shutil.copy('hds/cmp.h', 'include/star/')
    shutil.copy('hds/hds.h', 'include/star/')
    shutil.copy('hds/hds_fortran.h', 'include/')
    shutil.copy('hds/hds_fortran.h', 'include/star/')

    shutil.copy('chr/chr.h', 'include/')
    shutil.copy('hds-v4/hds_v4.h', 'include/star/')
    shutil.copy('hds-v5/hds_v5.h', 'include/star/')

    shutil.copy('starutil/util.h', 'include/star/')
