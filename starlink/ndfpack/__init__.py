#!/usr/bin/env python

"""
Python interface to the STARLINK ndf library

The STARLINK ndf library is used to access STARLINK's HDS files with
extension .sdf. This package enables read access to such files.
For instance, you can read in an ndf file using::

    import starlink.ndfpack

    ndf = starlink.ndfpack.Ndf('image')

See documentation on the Ndf class for how to access the various components.
Ndf is the main class and reads in entire Ndf files in a recursive manner.  In
the above example, ndf.data would contain the main data, ndf.head would
contain extension information (that can be extensive). This could cause memory
problems with very large files, but should at least allow fairly complete
access to all NDF components.

Classes
=======

* :class:`starlink.ndfpack.axis.Axis`  represents an NDF Axis component
* :class:`starlink.ndfpack.ndf.Ndf`    represents Starlink NDF files

Functions
=========

The :mod:`starlink.ndf` and :mod:`starlink.hds`
modules enable access to low-level
functions to access NDF components and HDS files.
Only use these if the standard slurp all
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

from starlink.ndfpack.axis import Axis
from starlink.ndfpack.ndf import Ndf
