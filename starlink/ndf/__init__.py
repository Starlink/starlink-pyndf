#!/usr/bin/env python

"""
Python interface to the STARLINK ndf library

The STARLINK ndf library is used to access STARLINK's HDS files with
extension .sdf. This package enables read access to such files.
For instance, you can read in an ndf file using:

import starlink.ndf

ndf = starlink.ndf.Ndf('image')

See documentation on the Ndf class for how to access the various components.
Ndf is the main class and reads in entire Ndf files in a recursive manner.  In
the above example, ndf.data would contain the main data, ndf.head would
contain extension information (that can be extensive). This could cause memory
problems with very large files, but should at least allow fairly complete
access to all NDF components. The module also enables access to low-level
functions to access ndf components. Only use these if you are familiar with
the standard ndf library. You may need a little bit of knowledge of NDF to be
fully confident with aspects of this module.

Classes
=======

Axis    -- represents an NDF Axis component
Ndf     -- represents Starlink NDF files

Functions
=========

There are many functional equivalents to NDF routines such as
dat_annul, dat_cell, dat_find, ndf_acget, ndf_aread and ndf_begin.
Look at end of documentation to see the list and refer to the NDF
documentation for their use. Only use these if the standard slurp all
information behaviour of Ndf is not what you want. You should not
normally need these low level routines.

License
=======

Copyright 2009-2011 Tom Marsh
All Rights Reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

from ._ndf import *
from .Axis import Axis
from .Ndf import Ndf

