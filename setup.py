
#    Copyright 2009-2011 Tom Marsh
#    Copyright 2011 Tim Jenness
#    All Rights Reserved.

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from numpy.distutils.core import setup, Extension
import numpy.distutils.fcompiler as fcompiler

import sys, os, subprocess, numpy

# Can not seem to work out how to get the --fcompiler option
# to work with numpy.distutils so look for it myself
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--fcompiler", dest="fcompiler",
                  help="Override default Fortran compiler")
(options,args) = parser.parse_args()


have_ast = 0
try:
    import starlink.Ast
    have_ast = 1
except:
    have_ast = 0
    print("")
    print("  Will not be building with AST facilities.")
    print("  Install starlink.Ast in order to read and write AST FrameSets.")
    print("")

""" Setup script for the ndf python extension"""

library_dirs = []
include_dirs = []

# need to direct to where includes and libraries are
# use starlink command ndf_link to generate library list. Need to strip off '-l' from each one
if 'STARLINK_DIR' in os.environ:
    os.environ['PATH'] = os.path.join(os.environ['STARLINK_DIR'], 'bin') + ':' + os.environ['PATH']
    libraries = subprocess.Popen(['ndf_link',''], shell=True, stdout=subprocess.PIPE, env=os.environ).communicate()[0].split()
    libraries = [x[2:] for x in libraries]
    # On python3 we want a unicode string
    if sys.version_info >= (3,):
        libraries = [ x.decode('ascii') for x in libraries]
    library_dirs.append(os.path.join(os.environ['STARLINK_DIR'], 'lib'))
    include_dirs.append(os.path.join(os.environ['STARLINK_DIR'], 'include'))
else:
    print("Environment variable STARLINK_DIR not defined!")
    exit(1)

include_dirs.append(numpy.get_include())

# NDF needs fortran runtime library for linking and HDS does still
# come with a small fortran dependency
fc = fcompiler.new_fcompiler(compiler=options.fcompiler)
fc.customize()
libraries.extend( fc.libraries )
library_dirs.extend( fc.library_dirs )

if have_ast:
    include_dirs.append(starlink.Ast.get_include())

ndf = Extension('starlink.ndf',
                define_macros        = [('MAJOR_VERSION', '0'),
                                        ('MINOR_VERSION', '3'),
                                        ('HAVE_AST', have_ast)],
                undef_macros         = ['USE_NUMARRAY'],
                include_dirs         = include_dirs,
                library_dirs         = library_dirs,
                runtime_library_dirs = library_dirs,
                libraries            = libraries,
                sources              = [os.path.join('starlink', 'ndf', 'ndf.c')]
                )

hds = Extension('starlink.hds',
                define_macros        = [('MAJOR_VERSION', '0'),
                                        ('MINOR_VERSION', '3')],
                undef_macros         = ['USE_NUMARRAY'],
                include_dirs         = include_dirs,
                library_dirs         = library_dirs,
                runtime_library_dirs = library_dirs,
                libraries            = libraries,
                sources              = [os.path.join('starlink', 'hds','hds.c')]
                )

setup(name='starlink-pyndf',
      version='0.3',
      packages =['starlink','starlink.ndfpack'],
      ext_modules=[ndf,hds],
      # metadata
      author='Tom Marsh',
      author_email='t.r.marsh@warwick.ac.uk',
      description='Python interface to Starlink ndf libraries',
      url='http://www.astro.warwick.ac.uk/',
      license="GNU GPL v3",
      )

