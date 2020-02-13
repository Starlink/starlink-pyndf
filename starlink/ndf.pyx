"""

Cython module wrapping some functions from the Starlink ndf C library.

Modelled after the existing NDF wrapper.
0. Method wrappers
SKIP [ ] ndf.init (redundant)
[x] ndf.begin
[x] ndf.end
[x] ndf.open
SKIP [ ] ndf.ndf_getbadpixval


1.: An NDF object with the following methods:
# read only accessors;
# Get settrs...
[x] title
[x] label
[x] units
[x] dim
[x] bound
[x] xnumb



# Also an NDFMapped object?

# Methods
LOWER PRIORITY?  axis ones -- maybe not do? acget, aform, anorm, aread, astat, amap

[x] indf.acget
[x] indf.aform
[x] indf.anorm
[x] indf.aread
[x] indf.astat
[x] indf.amap

[x] indf.annul
[x] indf.bound
** don't need**[ ] indf.cget
[x] indf.gtwcs
[x] indf.read
[x] indf.state
[x] indf.xloc
[x] indf.xname
[x] indf.xstat
[x] indf.new       ???
[x] indf.xnew
[x] indf.map

"""

import numpy as np
cimport numpy as cnp

cnp.import_array()

#from cpython.ref cimport PyObject

from libc.string cimport memcpy

from cpython.exc cimport PyErr_NewException, PyErr_SetString
from libc.stdint cimport uint32_t, int64_t
from libc.stdlib cimport free

from starlink import hds
from starlink cimport hds


cimport ndf as cndf

from libc.stdlib cimport free, malloc

from starlink import Ast


import warnings

cdef char* _allocate_cstring(length):
    cdef char* c_string = <char *> malloc(length * sizeof(char))
    if not c_string:
        raise MemoryError()
    return c_string

def init():
    """ Does nothing"""
    warnings.warn('ndf.init not longer does anything', DeprecationWarning)
    pass

def begin():
    """ Start a new NDF context.
    """
    cndf.ndfBegin()

def end():
    """
    End the current NDF context.
    """
    cdef int status=cndf.SAI__OK
    cndf.errBegin(&status)
    cndf.ndfEnd(&status)
    hds.raiseStarlinkException(status)


def open(filename, mode='READ', stat='OLD'):
    """
    Open an existing or new NDF file.

    args
    ----
    filename: str, path toNDF file on disk.

    kwargs
    ------
    mode: str, default='READ' ('UPDATE', 'READ' or 'WRITE')
    stat: str, default='OLD' ('OLD', 'NEW', or 'UNKNOWN')

    Return
    ------
    NdfWrapper object
    """

    if mode.upper() not in ('READ', 'UPDATE', 'WRITE'):
        raise hds.StarlinkError('Incorrect mode (%s) to open NDF: must be READ, WRITE or UPDATE' % mode)
    if stat.upper() not in ('OLD', 'NEW', 'UNKNOWN'):
        raise hds.StarlinkError('Incorrect status string for ndf_open )%s): must be OLD, NEW or UNKNOWN' %stat)

    filename = filename.encode()
    mode = mode.encode('ascii')
    stat = stat.encode('ascii')

    cdef int status = cndf.SAI__OK
    cndf.errBegin(&status)
    cdef int indf = cndf.NDF__NOID
    cdef int placeholder = cndf.NDF__NOPL
    cndf.ndfOpen( NULL, filename, mode, stat, &indf, &placeholder, &status)
    hds.raiseStarlinkException(status)

    return NDFWrapperClass.from_ints(indf, placeholder)


cdef _char_getter( int ndfid, char * component):
    cdef int state, status = cndf.SAI__OK

    # Check the state of the component.
    cndf.errBegin(&status)
    cndf.ndfState(ndfid, component, &state, &status)
    hds.raiseStarlinkException(status)

    # Return None if it does not exist
    if state==0:
        return None

    # Get the size of the component.
    cdef size_t clen
    cndf.errBegin(&status)
    cndf.ndfClen(ndfid, component, &clen, &status)
    hds.raiseStarlinkException(status)

    # Read the component into a C string
    cdef char* c_string = _allocate_cstring(clen+1)
    cndf.ndfCget(ndfid, component, &c_string[0], clen+1, &status)
    hds.raiseStarlinkException(status)

    py_string = c_string[:clen].decode()
    return py_string


cdef _char_setter(int ndfid, char *component, value):
    """
    If this doesn't work, it will try to set the component to an
    undefined state.
    """
    if value:
        value = value.encode('ascii')
    cdef int status = cndf.SAI__OK

    cndf.errBegin(&status)
    if value:
        cndf.ndfCput(value, ndfid, component, &status)
    else:
        cndf.ndfReset(ndfid, component, &status)
    hds.raiseStarlinkException(status)



cdef class NDFWrapperClass:
    """
    A wrapper class for the NDF C/C++ data structure.\

    As this is only using ints instead of pointers, its possible that
    this doesn't need to use a cdef file?

    Presumably requires that NDFBegin  has been created.
    """
    cdef int _ndfid
    cdef int _place


    @staticmethod
    cdef NDFWrapperClass from_ints(int _ndfid, int _place):
        cdef NDFWrapperClass wrapper = NDFWrapperClass.__new__(
            NDFWrapperClass)
        wrapper._ndfid = _ndfid
        wrapper._place = _place
        return wrapper


    @property
    def title(self):
        """Title of the NDF"""
        return _char_getter(self._ndfid, 'TITLE')

    @title.setter
    def title(self, title):
        _char_setter(self._ndfid, 'TITLE', title)

    @property
    def label(self):
        """Data label"""
        return _char_getter(self._ndfid, 'LABEL')
    @label.setter
    def label(self, label):
        _char_setter(self._ndfid, 'LABEL', label)

    @property
    def units(self):
        """Units of the data array"""
        return _char_getter(self._ndfid, 'UNITS')
    @units.setter
    def units(self, units):
        _char_setter(self._ndfid, 'UNITS', units)


    @property
    def dim(self):
        """ Dimensions of the NDF """
        cdef int status = cndf.SAI__OK
        cdef hds.hdsdim idim[cndf.NDF__MXDIM]
        cdef int ndim
        cndf.errBegin(&status)
        cndf.ndfDim(self._ndfid, cndf.NDF__MXDIM, idim, &ndim, &status)
        hds.raiseStarlinkException(status)

        dims = <list>idim
        return dims[0:ndim][::-1]

    @property
    def xnumb(self):
        """Number of extensions"""
        cdef int status = cndf.SAI__OK
        cdef int nextn = 0
        cndf.errBegin(&status)
        cndf.ndfXnumb(self._ndfid, &nextn, &status)
        hds.raiseStarlinkException(status)
        return nextn

    def aform(self, comp, iaxis):
        """Get the storage form of an axis object"""
        comp = comp.encode('ascii')
        cdef int status = cndf.SAI__OK
        naxis = len(self.ndim) - iaxis
        cdef char value[cndf.NDF__SZFRM]
        hds.errBegin(&status)
        cndf.ndfAform(self._ndfid, comp, naxis, value, cndf.NDF__SZFRM, &status)
        hds.raiseStarlinkException(status)
        return value.decode('ascii')

    def astat(self, comp, iaxis):
        """
        Determine the state of an NDF axis component

        Raises an error if axis component does not exist.
        """
        comp = comp.encode('ascii')
        naxis = len(self.ndfim) - iaxis
        cdef int status = cndf.SAI__OK
        cdef int state = 0
        hds.errBegin(&status)
        cndf.ndfAstat(self._ndfid, comp, naxis, &state, &status)
        hds.raiseStarlinkException(status)
        return bool(status)

    def cget(self, comp):
        """
        Get char component COMP.

        Returns None if comp does not exist
        """
        comp = comp.encode('ascii')
        return _char_getter(self._ndfid, comp)

    def acget(self, comp, iaxis):
        """
        Get char component COMP of axis iaxis

        args
        ----
        comp: str,
        iaxis: int (countstarts at 0, c ordering)

        returns
        -------
        None if comp does not exist
        """
        comp = comp.encode('ascii')
        cdef int status = SAI__OK
        cdef int state = 0
        cdef int naxis = len(self.dim) - iaxis

        hds.errBegin(&status)

        # Check if axis has defined values.
        cndf.ndfAstat(self._ndfid, comp, naxis, &state, &status)
        hds.raiseStarlinkException(status)
        if state == 0:
            return None
        cdef size_t clen
        cdef char * axisname = _allocate_cstring(clen + 1)

        # Get length of char string and name
        hds.errBegin(&status)
        cndf.ndfAclen(self._ndfid, comp, naxis, &clen, &status)
        cndf.ndfAcget(self._ndfid, comp, naxis, axisname, clen+1, &status)
        hds.raiseStarlinkException(status)

        return axisname.decode('ascii')

    def anorm(self, iaxis):
        """
        Get axis normalisation flag.

        if iaxis=-1 then the OR of all flags is returned.
        """
        cdef int state = 0
        cdef int status = cndf.SAI__OK
        if iaxis == -1:
            naxis = 0
        else:
            naxis = len(self.dim) - iaxis
        hds.errBegin(&status)
        ndfAnorm(self._ndfid, naxis, &state, &status)
        hds.raiseStarlinkException(status)
        return bool(state)


    def amap(self, comp, iaxis, type_, mmod):
        """Map an axis array component."""

        axis_comps = ('CENTRE', 'WIDTH', 'VARIANCE', 'ERROR')
        mapping_modes = ('READ', 'WRITE', 'UPDATE')
        data_types = ('_INTEGER', '_INT64', '_REAL', '_DOUBLE', '_WORD', '_UWORD', '_BYTE', '_UBYTE',
                      '_LOGICAL', '_CHAR', '_CHAR*')
        if mmod.upper() not in mapping_modes:
            raise hds.StarlinkWarning('Unsupported NDF axis mapping mode: must be one of {}'.format(
                mapping_modes))
        if comp.upper() not in axis_comps:
            raise hds.StarlinkWarning('Unsupported NDF axis data component: must be one of {}'.format(
                axis_comps))
        if type_.upper() not in data_types:
            raise hds.StarlinkWarning('Unsupported NDF data type: must be one of {}'.format(
                data_types))

        comp = comp.encode('ascii')
        type_ = type_.encode('ascii')
        mmod = mmod.encode('ascii')


        # Check the axis exists.
        cdef int status = hds.SAI__OK
        hds.errBegin(&status)

        cdef int state = 0
        cndf.ndfAstat(self._ndfid, comp, naxis, &state, &status)
        if state == 0:
            return None

        # Get the axis number and expected number elements in correct format.
        cdef int naxis = len(self.dim) - iaxis
        nelem = self.dim[naxis]


        # Map the data
        cdef void* ptr
        cdef size_t el = 0
        hds.errBegin(&status)
        cndf.ndfAmap(self._ndfid, comp, naxis, type_, b'READ', &ptr, &el, &status)

        # Raise an error if its not the correct size.
        hds.raiseStarlinkException(status)
        if el != nelem:
            raise hds.StarlinkException('ndfAmap: number of elements (%i) mapped different from number expected (%i)'
                                        %(el, nelem))
        return NDFMapped.from_pointer(self._ndfid, ptr, comp, type_, mmod, nelem, iaxis=iaxis)

    def aread(self, comp, iaxis):
        """
        Read component comp of axis iaxis.

        Return None if it doesn't exist
        """
        comp = comp.encode('ascii')
        cdef int status = hds.SAI__OK
        hds.errBegin(&status)
        cdef int naxis = len(self.dim) - iaxis
        cdef int state = 0
        cndf.ndfAstat(self._ndfid, comp, naxis, &state, &status)
        if state == 0:
            return None

        idim = self.dim
        ndim = len(self.dim)
        nelem = idim[naxis -1]

        #// Determine the data type
        cdef char type_[cndf.DAT__SZTYP+1]
        hds.errBegin(&status)
        cndf.ndfAtype(self._ndfid, comp, naxis, type_, cndf.DAT__SZTYP+1, &status)
        hds.raiseStarlinkException(status)
        cdef int np_type = hds._hdstype2numpy(type_[:])

        # Get the dimensions in correct form.
        cdef cnp.npy_intp pydim[1]
        pydim[0] = <cnp.npy_intp>nelem


        # Get the data.
        cdef cnp.ndarray arr
        cdef size_t nread
        arr = cnp.PyArray_SimpleNew(ndim, pydim, np_type)
        hds.errBegin(&status)
        cndf.ndfAmap(self._ndfid, comp, naxis, type_, b'READ', <void **>&arr.data, &nread, &status)
        hds.raiseStarlinkException(status)
        if nelem != nread:
            raise hds.StarlinkException('ndf_aread: number of elements (%i) different from number expected (%i)'
                                        % (nread, nelem))
        hds.errBegin(&status)
        cndf.ndfAunmp(self._ndfid, comp, naxis, &status)
        hds.raiseStarlinkException(status)
        return arr

    def new(self, type_, ndim, lbnd, ubnd):
        """
        Create a new simple NDF structure within an NDF.

        args
        ----
        type_: str, NDF primitive type.
        ndim: int, number of dimensions
        lbnd: int array, lower pixel bounds
        ubnd: int array, upper pixel bounds


        """
        type_ = type_.encode('ascii')
        cdef int status = cndf.SAI__OK

        cdef hds.hdsdim lower[cndf.NDF__MXDIM]
        cdef hds.hdsdim upper[cndf.NDF__MXDIM]
        for i in range(ndim):
            lower[i] = lbnd[i]
            upper[i] = ubnd[i]

        cndf.errBegin(&status)
        cndf.ndfNew(type_, ndim, lower, upper, &self._place, &self._ndfid, &status)
        hds.raiseStarlinkException(status)


    def map(self, comp, type_, mode):
        """Map access to an array component
        args
        ----
        comp: str, DATA, VARIANCE, QUALITY or ERROR
        type_: str, HDS/NDF primitive type
        mode: str, ('READ', 'WRITE', UPDATE not supported)

        Returns the mapped array object.
        """


        if comp.upper() not in ['DATA', 'VARIANCE', 'QUALITY','ERROR']:
            raise hds.StarlinkError('Cannot map component %s' % comp)
        if mode.upper() not in ['READ', 'WRITE', 'UPDATE']:
            raise hds.StarlinkError('Unsupported mode %s' % mode)

        if comp.upper()=='QUALITY' and type_.upper() != '_UBYTE':
            raise hds.StarlinkError('QUALITY component must be mapped as _UBYTE, not %s' % type_)

        comp = comp.encode('ascii')
        type_ = type_.encode('ascii')
        mode = mode.encode('ascii')

        cdef int status = cndf.SAI__OK
        cdef size_t nelem
        cdef void * ptr
        hds.errBegin(&status)
        cndf.ndfMap(self._ndfid, comp, type_, mode, &ptr, &nelem, &status)
        hds.raiseStarlinkException(status)

        return NDFMapped.from_pointer(self._ndfid, ptr, comp, type_, mode, nelem, iaxis=0)


    def annul(self):
        """Annul the NDF identifier"""
        cdef int status = cndf.SAI__OK
        cndf.errBegin(&status)
        cndf.ndfAnnul(&self._ndfid, &status)
        hds.raiseStarlinkException(status)

    def bound(self):
        """Return the pixel bounds of the NDF (2, ndim)"""

        cdef int status = cndf.SAI__OK
        cdef int ndim, i
        cdef hds.hdsdim ubnd[cndf.NDF__MXDIM]
        cdef hds.hdsdim lbnd[cndf.NDF__MXDIM]

        cndf.errBegin(&status)
        cndf.ndfBound(self._ndfid, cndf.NDF__MXDIM, &lbnd[0],
                      &ubnd[0],
                      &ndim, &status)

        pydims = []
        for i in range(0, ndim):
            pydims.append((lbnd[i], ubnd[i]))

        hds.raiseStarlinkException(status)
        return pydims


    def gtwcs(self):
        """Return the WCS as an AST frameset"""
        #print('Starting WCS')
        cdef cndf.AstFrameSet * wcs = NULL
        cdef int status  = cndf.SAI__OK
        cdef char * wcsstring
        #cdef  PyObject * pywcs

        return_obj = None
        cndf.errBegin(&status)
        cndf.astBegin
        cndf.ndfGtwcs(self._ndfid, &wcs, &status)


        if (wcs):

            wcsstring = cndf.astToString(<cndf.AstObject *>wcs)
            cndf.astAnnul(<cndf.AstObject *> wcs)
            pywcsstring = wcsstring[:].decode()
            cndf.astFree(wcsstring)
            chan = Ast.Channel(pywcsstring.split('\n'))
            return_obj = chan.read()
        cndf.astEnd
        hds.raiseStarlinkException(status)
        return return_obj


    def read(self, comp):
        """
        Read data from a component (DATA, VARIANCe or NOISE) of an NDF.

        Returns None if it doesn't exist?
        """

        comp = comp.encode('ascii')
        cdef int state, status = cndf.SAI__OK

        cndf.errBegin(&status)
        cndf.ndfState(self._ndfid, comp,
                      &state, &status)
        hds.raiseStarlinkException(status)
        # Return None if component does not exist.
        if state == 0:
            return None

        # Get number of dimensions  and shape of array
        cdef int ndim
        cdef cndf.hdsdim idim[cndf.NDF__MXDIM]


        cndf.errBegin(&status)
        cndf.ndfDim(self._ndfid, cndf.NDF__MXDIM, idim,
                    &ndim, &status)
        hds.raiseStarlinkException(status)

        # Determine the data type of the array
        cdef char type_[cndf.DAT__SZTYP+1]
        cndf.errBegin(&status)
        cndf.ndfType(self._ndfid, comp, type_, cndf.DAT__SZTYP+1, &status)
        hds.raiseStarlinkException(status)

        # Create array of correct dimensions and type.
        cdef int np_type = hds._hdstype2numpy(type_[:])
        #print('NP_TYPE is', np_type, 'HDS type is', type_)
        cdef cnp.npy_intp pydim[hds.DAT__MXDIM]

        cdef cnp.ndarray mypyarray
        for i in range(0, ndim):
            pydim[i] = idim[ndim -i -1]
        mypyarray = cnp.PyArray_SimpleNew(ndim, pydim, np_type)

        # Get byte size.
        cdef size_t nbyte = mypyarray.dtype.itemsize

        # Map data and check that number of elements == number of pixels
        cdef void * pntr[1]
        cdef size_t nelem, npix

        cndf.errBegin(&status)
        cndf.ndfSize(self._ndfid, &npix, &status)
        hds.raiseStarlinkException(status)

        cndf.errBegin(&status)
        cndf.ndfMap(self._ndfid, comp, type_, "READ",
                    pntr, &nelem, &status)
        hds.raiseStarlinkException(status)

        if nelem != npix:
            raise hds.StarlinkException('error reading NDF: number of elements differs from number expected')

        memcpy(mypyarray.data, pntr[0], npix * nbyte)
        cndf.errBegin(&status)
        cndf.ndfUnmap(self._ndfid, comp, &status)
        hds.raiseStarlinkException(status)
        return mypyarray

    def state(self, comp):
        """
        Determine the state of an NDF component (DATA/VAR).

        Returns TRUE or FALSE
        """
        comp = comp.encode('ascii')
        cdef int status, state = cndf.SAI__OK

        cndf.errBegin(&status)
        cndf.ndfState(self._ndfid, comp,
                      &state, &status)
        hds.raiseStarlinkException(status)
        return bool(state)

    def type(self, comp):
        """
        Determine the numeric type of an NDF compoentn.

        returns the NDF type, e.g. _INTEGER, _REAL etc.
        """
        comp = comp.encode('ascii')
        cdef char type[cndf.NDF__SZTYP+1]
        cdef int status = cndf.SAI__OK
        cndf.errBegin(&status)
        cndf.ndfType(self._ndfid, comp, type, cndf.NDF__SZTYP+1, &status)
        hds.raiseStarlinkException(status)
        return type.decode()

    def xname(self, nex):
        """
        Return name of NDF extension n.

        (counts from 0)
        """
        cdef int status = cndf.SAI__OK
        cdef char xname[cndf.NDF__SZXNM+1]
        cndf.errBegin(&status)
        cndf.ndfXname(self._ndfid, nex+1, xname, cndf.NDF__SZXNM+1, &status)
        hds.raiseStarlinkException(status)
        return xname.decode()

    def xloc(self, extname, mode):
        """
        Return an NDF extension as an HDS locator object.
        """
        extname = extname.encode('ascii')
        mode = mode.encode('ascii')

        cdef int status = cndf.SAI__OK
        cdef hds.HDSLoc * loc

        cndf.errBegin(&status)
        cndf.ndfXloc(self._ndfid, extname, mode, &loc, &status)
        hds.raiseStarlinkException(status)

        hds_wrapper = hds.HDSWrapperClass.from_pointer(loc)
        return hds_wrapper

    def xstat(self, extname):
        """Determine if extension xname exists"""
        extname = extname.encode('ascii')
        cdef int status = cndf.SAI__OK
        cdef int state
        cndf.errBegin(&status)
        cndf.ndfXstat(self._ndfid, extname, &state, &status)
        hds.raiseStarlinkException(status)
        return bool(state)

    def xdel(self, extname):
        """Delete extension extname.

        If extension does not exist, nothing is done"""
        extname = extname.encode('ascii')
        cdef int status = cndf.SAI__OK

        cndf.errBegin(&status)
        cndf.ndfXdel(self._ndfid, extname, &status)
        hds.raiseStarlinkException(status)

    def xnew(self, extname, type_, ndim=0, dim=None):
        """Create a new NDF extension

        Returns an HDS Loc to the new object.
        """

        extname = extname.encode('ascii')
        type_ = type_.encode('ascii')
        cdef int status = cndf.SAI__OK
        cdef hds.HDSLoc * loc = NULL;

        cdef hds.hdsdim dim_c[hds.DAT__MXDIM]

        # If it is an array rather than a structure or extension.
        if ndim == 0:
            cndf.errBegin(&status)
            cndf.ndfXnew(self._ndfid, extname, type_, 0, <hds.hdsdim*>0, &loc, &status)
            hds.raiseStarlinkException(status)

        else:
            if ndim < 1:
                raise SyntaxError('Number of dimensions must be 0 or greater in an NDF extension')
            if dim is None:
                raise SyntaxError('If number dimensions!=0, then shape (dim) must be provided')

            for i in range(0,ndim):
                dim_c[i] = dim[i]

            cndf.errBegin(&status)
            cndf.ndfXnew(self._ndfid, extname, type, ndim, dim_c, &loc, &status)
            hds.raiseStarlinkException(status)
        return  hds.HDSWrapperClass.from_pointer(loc)



# A class for NDF Mapped objects (is this really necessary???)
cdef class NDFMapped:
    """
    Wrapper for the pointer to an NDF mapped component.
    """

    cdef void * _pntrobject
    cdef int _ndfid
    cdef readonly char _comp[hds.DAT__SZNAM+1]
    cdef readonly char _type[cndf.NDF__SZTYP+1]
    cdef readonly char _mode[cndf.NDF__SZMMD+1]
    cdef readonly int nelem
    cdef readonly int iaxis


    def __dealloc__(self):
        """
        This has to catch a starlink exception, for those cases where
        ndf.end() or indf.annul() is called before the ndfmap object
        has been deleted.
        """
        cdef int status, valid = cndf.SAI__OK
        cndf.ndfValid(self._ndfid, &valid, &status)
        try:
            hds.raiseStarlinkException(status)
        except hds.StarlinkError:
            pass
        if valid:
            self._unmap()

    def _unmap(self):
        cdef int status = cndf.SAI__OK
        hds.errBegin(&status)
        comp = self.comp.encode('ASCII')
        if self.iaxis == 0:
            cndf.ndfUnmap(self._ndfid, comp, &status)
        else:
            cndf.ndfAunmp(self._ndfid, comp, self.iaxis, &status)
        hds.raiseStarlinkException(status)

    def numpytondf(self, newdata):
        nptype = hds._hdstype2numpy(self._type)
        cdef cnp.ndarray npyval
        cdef void *ptr = self._pntrobject

        # This includes requirement that it is contiguous c layout.
        cdef int requirements = (cnp.NPY_ARRAY_CARRAY | cnp.NPY_ARRAY_FORCECAST)
        npyval = cnp.PyArray_FROMANY(newdata, nptype, 0, hds.DAT__MXDIM, requirements)

        # Handle some possible errors
        if npyval is None:
            raise hds.StarlinkError('Could not create array for data')
        if self.mode is 'READ':
            raise hds.StarlinkError('Array was mapped for reading: cannot update values')
        if npyval.size != self.nelem:
            raise hds.StarlinkError('Number of elements in np array (%i) different from number of elements mapped (%i)' %(npyval.size), self.nelem))

        # Get the size of each byte.
        cdef size_t nbyte = npyval.dtype.itemsize
        memcpy(ptr, &npyval.data[0], self.nelem *nbyte)


    @staticmethod
    cdef NDFMapped from_pointer(int ndfid, void* pntrobject, char * comp, char * type_,
                                char * mode, int nelem, iaxis=0):
        """
        Factory function to create NDFMapped objects from pointer.
        """
        cdef NDFMapped mapped = NDFMapped.__new__(NDFMapped)

        mapped._pntrobject = pntrobject
        mapped._ndfid = ndfid
        mapped._comp = comp
        mapped._mode = mode
        mapped._type = type_
        mapped.nelem = nelem
        mapped.iaxis = iaxis
        return mapped

    @property
    def comp(self):
        return self._comp.decode()

    @property
    def mode(self):
        return self._mode.decode()

    @property
    def type(self):
        return self._type.decode()







