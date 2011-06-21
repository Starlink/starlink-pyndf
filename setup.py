from distutils.core import setup, Extension

import sys, os, subprocess, numpy

""" Setup script for the ndf python extension"""

library_dirs = []
include_dirs = []

# need to direct to where includes and libraries are
# use starlink command ndf_link to generate library list. Need to strip off '-l' from each one
if 'STARLINK_DIR' in os.environ:
    os.environ['PATH'] = os.path.join(os.environ['STARLINK_DIR'], 'bin') + ':' + os.environ['PATH']
    libraries = subprocess.Popen(['ndf_link',''], shell=True, stdout=subprocess.PIPE, env=os.environ).communicate()[0].split()
    libraries = [x[2:].decode('ascii') for x in libraries]
    library_dirs.append(os.path.join(os.environ['STARLINK_DIR'], 'lib'))
    include_dirs.append(os.path.join(os.environ['STARLINK_DIR'], 'include'))
else:
    print("Environment variable STARLINK_DIR not defined!")
    exit(1)

include_dirs.append(numpy.get_include())

ndf = Extension('trm.ndf._ndf',
                define_macros        = [('MAJOR_VERSION', '0'),
                                        ('MINOR_VERSION', '1')],
                undef_macros         = ['USE_NUMARRAY'],
                include_dirs         = include_dirs,
                library_dirs         = library_dirs,
                runtime_library_dirs = library_dirs,
                libraries            = libraries,
                sources              = [os.path.join('trm', 'ndf', 'ndf.c')]
                )

setup(name='trm.ndf',
      version='0.1',
      packages =['trm','trm.ndf'],
      ext_modules=[ndf],

      # metadata
      author='Tom Marsh',
      author_email='t.r.marsh@warwick.ac.uk',
      description='Python interface to ndf libraries',
      url='http://www.astro.warwick.ac.uk/',

      )

