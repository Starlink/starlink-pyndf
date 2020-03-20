import starlink.ndf as ndf
import starlink.hds as hds
from starlink.ndfpack.axis import Axis

import re
import numpy as n


class Ndf(object):
    """
    Represents Starlink NDF files

    Attributes may not all be defined.

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
        Python as in::

            ndf = starlink.ndfpack.Ndf('image')
            subim = image.data[0:5,0:4,0:3]
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
        ndf.init()
        ndf.begin()
        try:
            indf = ndf.open(fname)
            #: the data array, a numpy N-d array
            self.data  = indf.read('Dat')
            #: pixel limits of data array.
            #: 2xndim array of lower and upper bounds
            self.bound = indf.bound()
            #: variances, a numpy N-d array
            self.var   = indf.read('Var')
            #: label string associated with the data
            self.label = indf.label
            #: title string associated with the data
            self.title = indf.title
            #: data unit string
            self.units  = indf.units
            try:
                #: WCS information, a PyAST FrameSet
                self.wcs = indf.gtwcs()
            except NotImplementedError:
                self.wcs = None

            # Read the axes
            #: a list of Axis objects, one for each dimension of data
            self.axes = []
            for nax in range(self.data.ndim):
                self.axes.append(Axis(indf, nax))

            # Read the extensions
            #: header/extensions, a dictionary
            self.head = {}
            nextn = indf.xnumb
            for nex in range(nextn):
                xname = indf.xname(nex)
                loc1 = indf.xloc(xname, 'READ')
                hdsloc = hds._transfer(loc1)
                _read_hds(hdsloc, self.head)
                hdsloc.annul()

            ndf.end()
        except:
            ndf.end()
            raise

def _read_hds(loc, head, array=False):
    """Recursive reader of an HDS starting from locator = loc"""

    name = loc.name
    if loc.struc:
        dims = loc.shape
        if dims != None:
            head[name] = _create_md_struc(dims)
            sub = n.zeros(len(dims), n.int32)
            _read_md_struc(head[name], loc, dims, sub)
        else:
            if array:
                h = head
            else:
                h = head[name] = {}
            ncomp = loc.ncomp
            for ncmp in range(ncomp):
                loc1 = loc.index(ncmp)
                _read_hds(loc1, h, array)
                loc1.annul()
    elif loc.state:
        head[name] = loc.get()

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
        loc1 = loc.cell(sub)
        _read_hds(loc1, mds, True)
        loc1.annul()

        # update index array for next element
        sub[-1] += 1
        for i in range(len(sub)-1,0,-1):
            if sub[i] == dims[i]:
                sub[i]    = 0
                sub[i-1] += 1

