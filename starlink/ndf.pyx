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
[ ] NdfObj.title
[ ] NdfObj.label
[ ] .units
[ ] .xnumb


# Also an NDFMapped object?

# Methods
LOWER PRIORITY?  axis ones -- maybe not do? acget, aform, anorm, aread, astat, amap

[x] indf.annul
[x] indf.bound
[ ] indf.cget
[x] indf.gtwcs
[x] indf.read
[ ] indf.state
[ ] indf.xloc
[ ] indf.xname
[ ] indf.xstat
[ ] indf.new       ???
[ ] indf.xnew
[ ] indf.map

"""

#cimport mytesthds as hds
#import mytesthds as hds

import numpy as np
cimport numpy as cnp

cnp.import_array()

#from cpython.ref cimport PyObject

from libc.string cimport memcpy

from cpython.exc cimport PyErr_NewException, PyErr_SetString
from libc.stdint cimport uint32_t, int64_t
from libc.stdlib cimport free

from starlink import mytesthds as tthds
from starlink cimport mytesthds as thds

cimport myndf as cndf

from libc.stdlib cimport free, malloc


from starlink import Ast

cdef char* _allocate_cstring(length):
    cdef char* c_string = <char *> malloc(length * sizeof(char))
    if not c_string:
        raise MemoryError()
    return c_string


def begin():
    """ Start a new NDF context.

    COMMENT: I have no reall idea what that means...
    """
    cndf.ndfBegin()

def end():
    """
    End the current NDF context.
    """
    cdef int status=cndf.SAI__OK
    cndf.errBegin(&status)
    cndf.ndfEnd(&status)
    thds.raiseStarlinkException(status)


def open(filename, mode='READ', stat='OLD'):
    """
    Open an existing NDF file.

    args
    ----
    filename: str, path toNDF file on disk.

    kwargs
    ------
    mode: str, default='READ' ('UPDATE', 'READ'or 'WRITE')
    stat

    Return
    ------
    NdfWrapper object
    """

    if mode.upper() not in ('READ', 'UPDATE', 'WRITE'):
        raise tthds.StarlinkError('Incorrect mode (%s) to open NDF: must be READ, WRITE or UPDATE' % mode)
    if stat.upper() not in ('OLD', 'NEW', 'UNKNOWN'):
        raise tthds.StarlinkError('Incorrect status string for ndf_open )%s): must be OLD, NEW or UNKNOWN' %stat)

    filename = filename.encode()
    mode = mode.encode('ascii')
    stat = stat.encode('ascii')

    cdef int status = cndf.SAI__OK
    cndf.errBegin(&status)
    cdef int indf = cndf.NDF__NOID
    cdef int placeholder = cndf.NDF__NOPL
    cndf.ndfOpen( NULL, filename, mode, stat, &indf, &placeholder, &status)
    thds.raiseStarlinkException(status)

    return NDFWrapperClass.from_ints(indf, placeholder)


cdef _char_getter( int ndfid, char * component):
    cdef int state, status = cndf.SAI__OK

    # Check the state of the component.
    cndf.errBegin(&status)
    cndf.ndfState(ndfid, component, &state, &status)
    thds.raiseStarlinkException(status)

    # Get the size of the component.
    cdef size_t clen
    cndf.errBegin(&status)
    cndf.ndfClen(ndfid, component, &clen, &status)
    thds.raiseStarlinkException(status)

    # Read the component into a C string
    cdef char* c_string = _allocate_cstring(clen+1)
    cndf.ndfCget(ndfid, component, &c_string[0], clen+1, &status)
    thds.raiseStarlinkException(status)

    py_string = c_string[:clen].decode()
    return py_string

cdef _char_setter(int ndfid, char *component, value):
    """
    If this doesn't work, it will try to set the component to an
    undefined state.
    """

    cdef int status = cndf.SAI__OK

    cndf.errBegin(&status)
    if value:
        cndf.ndfCput(value, ndfid, component, &status)
    else:
        cndf.ndfReset(ndfid, component, &status)
    thds.raiseStarlinkException(status)



cdef class NDFWrapperClass:
    """
    A wrapper class for the NDF C/C++ data structure.\

    As this is only using ints instead of pointers, its possible that
    this doesn't need to use a cdef file?

    Presumably requires that NDFBegin  has been created.
    """
    cdef int _ndfid
    cdef int _place


    def __cinit__(self):

        print('Ran cinit for NDFWrapperClass!')



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
        """Title of the NDF"""
        return _char_getter(self._ndfid, 'LABEL')
    @label.setter
    def label(self, label):
        _char_setter(self._ndfid, 'LABEL', label)

    @property
    def units(self):
        """Title of the NDF"""
        return _char_getter(self._ndfid, 'UNITS')
    @units.setter
    def units(self, units):
        _char_setter(self._ndfid, 'UNITS', units)


    def annul(self):
        """Annul the NDF"""
        cdef int status = cndf.SAI__OK
        cndf.ndfAnnul(&self._ndfid, &status)
        thds.raiseStarlinkException(status)

    def bound(self):
        """Return the pixel bounds of the NDF"""

        cdef int status = cndf.SAI__OK
        cdef int ndim
        cdef int i


        cdef thds.hdsdim ubnd[cndf.NDF__MXDIM]
        cdef thds.hdsdim lbnd[cndf.NDF__MXDIM]



        cndf.errBegin(&status)
        cndf.ndfBound(self._ndfid, cndf.NDF__MXDIM, &lbnd[0],
                      &ubnd[0],
                      &ndim, &status)

        print('NDIM IS', ndim)
        print('NDF__MXDIM IS', cndf.NDF__MXDIM)

        pydims = []
        for i in range(0, ndim):
            print(lbnd[i], ubnd[i])
            pydims.append((lbnd[i], ubnd[i]))

        #bound = np.zeros((2,ndim), dtype=np.int64)
        #for i in range(0, ndim):
        #    bound[0,i] = lbnd[ndim -i -1]
        #    bound[1,i] = ubnd[ndim -i -1]
        thds.raiseStarlinkException(status)
        return pydims


    def gtwcs(self):
        """Return the WCS as an AST frameset"""
        print('Starting WCS')
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
        thds.raiseStarlinkException(status)
        return return_obj


    def read(self, comp):
        """
        Read data from a component (DATA, VARIANCe or NOISE) of an NDF.

        Returns None if it doesn't exist?
        """

        comp = comp.encode('ascii')
        cdef int status, state = cndf.SAI__OK

        cndf.errBegin(&status)
        cndf.ndfState(self._ndfid, comp,
                      &state, &status)
        thds.raiseStarlinkException(status)
        # Return None if component does not exist.
        if state == 0:
            return None

        # Get number of dimensions  and shape of array
        cdef int ndim
        cdef cndf.hdsdim idim[cndf.NDF__MXDIM]


        cndf.errBegin(&status)
        cndf.ndfDim(self._ndfid, cndf.NDF__MXDIM, idim,
                    &ndim, &status)
        thds.raiseStarlinkException(status)

        # Determine the data type of the array
        cdef char type_[cndf.DAT__SZTYP+1]
        cndf.errBegin(&status)
        cndf.ndfType(self._ndfid, comp, type_, cndf.DAT__SZTYP+1, &status)
        thds.raiseStarlinkException(status)

        # Create array of correct dimensions and type.
        cdef int np_type = thds._hdstype2numpy(type_[:])
        print('NP_TYPE is', np_type, 'HDS type is', type_)
        cdef cnp.npy_intp pydim[thds.DAT__MXDIM]

        cdef cnp.ndarray mypyarray
        for i in range(0, ndim):
            pydim[i] = idim[ndim -i -1]
        mypyarray = cnp.PyArray_SimpleNew(ndim, pydim, np_type)


        cdef size_t nbyte = mypyarray.dtype.itemsize
        print('NEW NBYTE IS', nbyte)
        # Map data and check that number of elements == number of pixels
        cdef void * pntr[1]
        cdef size_t nelem, npix

        cndf.errBegin(&status)
        cndf.ndfSize(self._ndfid, &npix, &status)
        thds.raiseStarlinkException(status)

        cndf.errBegin(&status)
        cndf.ndfMap(self._ndfid, comp, type_, "READ",
                    pntr, &nelem, &status)
        thds.raiseStarlinkException(status)
        print(type_[:])
        if nelem != npix:
            raise Exception('error reading NDF: number of elements differs from number expected')

        memcpy(mypyarray.data, pntr[0], npix * nbyte)
        cndf.errBegin(&status)
        cndf.ndfUnmap(self._ndfid, comp, &status)
        thds.raiseStarlinkException(status)

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
        thds.raiseStarlinkException(status)
        return bool(state)



# Methods
#dim
#title
#label
#units
#xnum
#LOWER PRIORITY?  axis ones -- maybe not do? acget, aform, anorm, aread, astat, amap

#[ ] indf.annul
#[ ] indf.bound
#[ ] indf.cget
#[ ] indf.gtwcs
#[ ] indf.read
#[ ] indf.state
#[ ] indf.xloc
#[ ] indf.xname
#[ ] indf.xstat
#[ ] indf.new       ???
#[ ] indf.xnew
#[ ] indf.map



    def bound_test(self):

        """Return the pixel bounds of the NDF"""

        cdef int status = cndf.SAI__OK
        cdef int ndim
        cdef int i

        cdef int _ndfid = <int>self._ndfid

        cdef thds.hdsdim  ubnd[cndf.NDF__MXDIM]
        cdef thds.hdsdim* lbnd = <thds.hdsdim *> malloc(cndf.NDF__MXDIM * sizeof(thds.hdsdim*))


        cndf.errBegin(&status)
        cndf.ndfBound(_ndfid, cndf.NDF__MXDIM, &lbnd[0], ubnd, &ndim, &status)
        thds.raiseStarlinkException(status)


        print('NDIM IS', ndim)
        print('NDF__MXDIM IS', cndf.NDF__MXDIM)

        pydims = []
        for i in range(0, ndim):
            pydims.append((lbnd[i], ubnd[i]))

        #bound = np.zeros((2,ndim), dtype=np.int64)
        #for i in range(0, ndim):
        #    bound[0,i] = lbnd[ndim -i -1]
        #    bound[1,i] = ubnd[ndim -i -1]

        return pydims
