#!/usr/bin/env python

"""
Python interface to the STARLINK ndf library

The STARLINK ndf library is used to access STARLINK's HDS files with
extension .sdf. This package enables read access to such files.
For instance, you can read in an ndf file using:

import trm.ndf

ndf = trm.ndf.Ndf('image')

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
"""

from ._ndf import *

import re
import numpy as n

class Axis(object):
    """
    Represents an NDF axis.

    Attributes (not all of which are guaranteed to be defined)

    pos    -- positions of centres of pixels
    var    -- variances of positions of centres of pixels
    width  -- widths of pixels
    label  -- character string label.
    units  -- units of the axis
    """

    def __init__(self, indf, iaxis):
        """Initialise an NDF axis."""
        self.pos   = ndf_aread(indf,'Centre',iaxis)
        self.var   = ndf_aread(indf,'Variance',iaxis)
        self.width = ndf_aread(indf,'Width',iaxis)
        self.label = ndf_acget(indf,'Label',iaxis)
        self.units = ndf_acget(indf,'Units',iaxis)

class Ndf(object):
    """
    Represents Starlink NDF files

    Attributes (not all of which may be defined):

    data  -- numpy array containing the maps
    var   -- variances
    axes  -- a list of Axis object, one for each dimension of data
    label -- label associated with the data
    title -- title associated with the data
    head  -- dictionary of header information

    Complete information on NDFs can be obtained from sun33 of the Starlink documentation
    and may illuminate the meaning of some of these.
    """

    def __init__(self, fname):
        """
        Initialise an NDF from a file.

        This slurps the whole thing in, including all extensions, axes etc. This could
        cause memory problems on large files. You can use either standard format
        NDF sections in such case or Pythonic ones. e.g. Given an NDF 'image' listed by
        hdstrace to have a data array DATA(3,4,5), the entire image can be specified
        using any of 'image', 'image(1:3,1:4,1:5)' or 'image[0:5,0:4,0:3]' where as usual
        with the index ranges in Python, the last index is NOT included and indices start
        at 0. While those used to dealing with NDFs may find the first with () more
        familiar, the second may be simpler if you deal with the outout from a python script
        since it is identically ordered and more consistent with taking sub-sections using
        Python as in

        ndf = trm.ndf.Ndf('image')
        subim = image.data[0:5,0:4,0:3]

        The following attributes are created:

        data    -- the data array, a numpy N-d array
        bound   -- pixel limits of data array. 2xndim array of lower and upper bounds
        var     -- variances, a numpy N-d array
        axes    -- Axis components
        label   -- label string
        title   -- title string
        head    -- header/extensions, a dictionary
        """
        object.__init__(self)

        # Next section changes from a pseudo-Pythonic version of an NDF section
        # to a Fortran-like one i.e. reverse the indices, add 1 to the first of a pair
        # or to the sole index
        reg = re.compile(r'([^\[\]]*)\[([^\[\]]*)\]')
        m = reg.match(fname)
        if m != None:
            tup = m.group(2).split(',')
            nname = m.group(1) + '('
            for st in tup[-1:0:-1]:
                subt = st.split(':')
                if len(subt) == 1:
                    add = str(int(subt[0])+1)
                elif len(subt) == 2:
                    add = str(int(subt[0])+1) + ':' + str(int(subt[1]))
                else:
                    raise Exception('Could not understand ' + fname)
                nname += add + ','
            subt = tup[0].split(':')
            if len(subt) == 1:
                add = str(int(subt[0])+1)
            elif len(subt) == 2:
                add = str(int(subt[0])+1) + ':' + str(int(subt[1]))
            else:
                raise Exception('Could not understand ' + nname)
            nname += add + ')'
            fname = nname

        # OK, get on with NDF stuff
        ndf_init()
        ndf_begin()
        try:
            (indf,place) = ndf_open(fname)
            self.data  = ndf_read(indf,'Dat')
            self.bound = ndf_bound(indf)
            self.var   = ndf_read(indf,'Var')
            self.label = ndf_cget(indf,'Label')
            self.title = ndf_cget(indf,'Title')

            # Read the axes
            self.axes = []
            for nax in range(self.data.ndim):
                self.axes.append(Axis(indf, nax))

            # Read the extensions
            self.head = {}
            nextn = ndf_xnumb(indf)
            for nex in range(nextn):
                xname = ndf_xname(indf, nex)
                loc1 = ndf_xloc(indf, xname, 'READ')
                _read_hds(loc1, self.head)
                dat_annul(loc1)

            ndf_end()
        except:
            ndf_end()
            raise

def _read_hds(loc, head, array=False):
    """Recursive reader of an HDS starting from locator = loc"""

    name = dat_name(loc)
    if dat_struc(loc):
        dims = dat_shape(loc)
        if dims != None:
            head[name] = _create_md_struc(dims)
            sub = n.zeros(dims.size, int)
            _read_md_struc(head[name], loc, dims, sub)
        else:
            if array:
                h = head
            else:
                h = head[name] = {}
            ncomp = dat_ncomp(loc)
            for ncmp in range(ncomp):
                loc1 = dat_index(loc, ncmp)
                _read_hds(loc1, h, array)
                dat_annul(loc1)
    elif dat_state(loc):
        head[name] = dat_get(loc)

def _create_md_struc(dims):
    """Creates a multi-dimensional list of dictionaries to represent multi-dimensional structures"""
    if len(dims) > 1:
        ndims = dims[1:]
        return [_create_md_struc(ndims) for i in range(dims[0])]
    else:
        return [{} for i in range(dims[0])]

def _read_md_struc(mds, loc, dims, sub):
    """Recursive reader of a structure array pointed to by loc. sub must start at (0,0,...,0)"""
    if isinstance(mds, list):
        for mdst in mds:
            _read_md_struc(mdst, loc, dims, sub)
    else:
        loc1 = dat_cell(loc, sub)
        _read_hds(loc1, mds, True)
        dat_annul(loc1)

        # update index array for next element
        sub[-1] += 1
        for i in range(len(sub)-1,0,-1):
            if sub[i] == dims[i]:
                sub[i]    = 0
                sub[i-1] += 1

