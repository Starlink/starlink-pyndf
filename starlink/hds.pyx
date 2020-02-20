from libc.stdlib cimport free, malloc
"""

Cython module wrapping some functions from the Starlink hds C library.

"""

cimport starlink.hds as chds
from cpython.exc cimport PyErr_NewException, PyErr_SetString
from libc.stdint cimport uint32_t, int64_t
from libc.stdlib cimport free

cimport numpy as cnp
import numpy as np
cnp.import_array()



class StarlinkError(Exception):
    pass

cdef int raiseStarlinkException( int status ) except *:
    """ flush errors and close error context.

    errBegin must have been called before this.
    """
    # if no errors, end status and return 0
    if status == chds.SAI__OK:
        chds.errEnd(&status);
        return 0;

    # Otherwise we want to get the full error message. Each call to
    # chds.errLoad will set param to the error message name, and opstr
    # to the error message itself. The status will be set to SAI__OK
    # when no more error messages.
    cdef int errorstatus = 0;
    cdef char param[chds.ERR__SZPAR+1];
    cdef char opstr[chds.ERR__SZMSG+1];
    cdef int parlen = 0;
    cdef int oplen = 0;

    errormessages = [];
    while errorstatus != chds.SAI__OK:
        chds.errLoad(param, sizeof(param), &parlen, opstr, sizeof(opstr), &oplen, &errorstatus)
        if errorstatus != chds.SAI__OK:
            error_name = param[:parlen]
            error_msg = opstr[:oplen]
            errormessages += ['%s: %s (status=%i)'% (error_name, error_msg, errorstatus)]

    chds.errEnd(&status)
    errormessage = '\n'.join(errormessages)
    print(errormessage)
    raise StarlinkError(errormessage)


cdef  int _hdstype2numpy( const char * type):
    cdef int retval
    if type==b"_INTEGER":
        retval=cnp.NPY_INT
    elif type==b"_INT64":
        retval=cnp.NPY_INT64
    elif type==b"_REAL":
        retval=cnp.NPY_FLOAT
    elif type==b"_DOUBLE":
        retval=cnp.NPY_DOUBLE
    elif type==b"_WORD":
        retval=cnp.NPY_INT16
    elif type==b"_UWORD":
        retval=cnp.NPY_USHORT
    elif type==b"_BYTE":
        retval=cnp.NPY_BYTE
    elif type==b"_UBYTE":
        retval=cnp.NPY_UBYTE
    elif type==b"_LOGICAL":
        retval=cnp.NPY_INT
    elif type[0:6]==b"_CHAR*":
        retval=cnp.NPY_STRING
    else:
        raise StarlinkError('Unknown HDS type %s cannot be converted to numpy values'%type.decode())
    return retval



def getbadvalue(typestr):
    """
    Returns bad pixel value for given HDS numerical type.
    """
    badvalue = None
    if typestr == "_DOUBLE":
        badvalue = chds.VAL__BADD
    elif typestr == "_REAL":
        badvalue = chds.VAL__BADR
    elif typestr == "_INTEGER":
        badvalue = chds.VAL__BADI
    elif typestr == "_INT64":
        badvalue = <int64_t>chds.VAL__BADK
    elif typestr == "_BYTE":
        badvalue = chds.VAL__BADB
    elif typestr == "_UBYTE":
        badvalue = chds.VAL__BADUB
    elif typestr == "_WORD":
        badvalue = chds.VAL__BADW
    elif typestr == "_UWORD":
        badvalue = chds.VAL__BADUW
    else:
        raise ValueError("Unknown Type %s; no known bad value for this type." % typestr)

    return badvalue

def open(filename, mode):
    """
    Open an existing HDS file.

    Args:
    -----
    filename: (str) name on disk for existing file.
    mode: (str): one of READ, WRITE or UPDATE

    Return:
    -------
    returns an HDSWrapper around the HDS Locator
    """
    filename = filename.encode()
    mode = mode.encode('ascii')
    cdef int status = chds.SAI__OK
    cdef chds.HDSLoc * loc = NULL
    chds.errBegin(&status);
    chds.hdsOpen(filename, mode, &loc, &status);
    raiseStarlinkException(status)
    return HDSWrapperClass.from_pointer(loc, owner=1)


def new(filename, hdsname, hdstype, dims=None):
    """
    Create a new HDS file.

    Args:
    -----
    filename: (str): name on disk for new file.
    hdsname: (str) name of HDS object
    hdstype: (str) type of HDS object (e.g. NDF, _DOUBLE, ...)

    Optional Kwargs:
    ----------------
    dims: if given, implies scalar object.

    Returns
    -------
    Returns a locator for a new HDS object.
    """

    # Ensure we have bytes?
    filename = filename.encode()
    hdsname = hdsname.encode('ascii')
    hdstype = hdstype.encode('ascii')

    cdef int status = chds.SAI__OK
    cdef chds.HDSLoc * outloc = NULL;

    # Handl dimensions
    cdef int ndim=0
    cdef chds.hdsdim cdims[chds.DAT__MXDIM]

    chds.errBegin(&status)

    cdef int i
    if dims:
        ndim = len(dims)
        for i in range(0, ndim):
            cdims[i] = dims[ndim -i -1]

    chds.hdsNew(filename, hdsname, hdstype, ndim, cdims, &outloc, &status)

    raiseStarlinkException(status)
    return HDSWrapperClass.from_pointer(outloc, owner=1)


# Declare a class of HDS objects. Note that python class constructor
# objects can't take pointers (or any non-convertible C object), so
# use the instantiation from existing C/C++ pointers from the cython
# documentation at
# https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#existing-pointers-instantiation
# This involves using a factory function.

# To create a new instance, you would use:
# hdsobj = HDSWrapperClass.from_pointer(HDSLoc* locator_pointer)

def _transfer(loc):
    """ This no longer needs to do anything."""
    return loc



cdef class HDSWrapperClass:
    """
    A wrapper class for the HDSLoc C/C++ data structure
    """

    def __cinit__(self):
        #Not the owner of this pointer
        self.ptr_owner = False

    def __dealloc__(self):
        # run the hds dealloc stuff here?
        if self._locator is not NULL and self.ptr_owner is True:
            free(self._locator)
            self._locator = NULL

    # Properties of the extension class
    # clen, name, ncomp, shape, state, struc, type and valid
    @property
    def name(self):
        """HDS component name"""

        cdef int status = chds.SAI__OK
        cdef char name_str[chds.DAT__SZNAM+1]

        chds.errBegin(&status)
        chds.datName(self._locator, name_str, &status);
        raiseStarlinkException(status)

        # Convert char to python string?
        return name_str.decode()

    @property
    def clen(self):
        """
        HDS component character string Length

        From original code:

          The routine returns the number of characters required to
          represent the values of a primitive object. If the object is
          character-type, then its length is returned
          directly. Otherwise, the value returned is the number of
          characters required to format the object's values (as a
          decimal string if appropriate) without loss of information.
        """

        cdef int status = chds.SAI__OK
        cdef size_t clen=0;

        chds.errBegin(&status)
        chds.datClen(self._locator, &clen, &status);
        raiseStarlinkException(status)

        return clen



    @property
    def ncomp(self):
        """
        Number of components in HDS structure

        If this is a scalar, ncomp will raise a StarlinkHDSError
        """

        cdef int status = chds.SAI__OK
        cdef int ncomp = 0
        chds.errBegin(&status)
        chds.datNcomp(self._locator, &ncomp, &status);
        raiseStarlinkException(status)

        return ncomp

    @property
    def shape(self):
        """Shape of component in HDS structure

        Will return None if it is a scalar.
        """

        cdef int status = chds.SAI__OK
        cdef chds.hdsdim tdim[chds.DAT__MXDIM]
        cdef int ndim
        chds.errBegin(&status)
        chds.datShape(self._locator, chds.DAT__MXDIM, tdim, &ndim, &status);
        raiseStarlinkException(status)
        if ndim == 0:
            return None

        pydims = []
        cdef int i
        for i in range(0, ndim):
            pydims.append(tdim[i])
        return pydims[::-1]

    @property
    def state(self):
        """The state of the HDS component

        Enquire the state of a primitive, ie. whether its value is defined or not.
        """
        cdef int status = chds.SAI__OK
        cdef int state;
        chds.errBegin(&status)
        chds.datState(self._locator, &state, &status);
        raiseStarlinkException(status)

        return bool(state)
    @property
    def struc(self):
        """Is the component a structure?"""

        cdef int status = chds.SAI__OK
        cdef int struc;
        chds.errBegin(&status)
        chds.datStruc(self._locator, &struc, &status);
        raiseStarlinkException(status)

        return bool(struc)
    @property
    def type(self):
        """Type of the HDS component."""
        cdef int status = chds.SAI__OK
        cdef char type_str[chds.DAT__SZTYP+1];

        chds.errBegin(&status)
        chds.datType(self._locator, type_str, &status);
        raiseStarlinkException(status)

        return type_str.decode()
    @property
    def valid(self):
        """Is the locator valid?"""

        cdef int status = chds.SAI__OK
        cdef int valid;
        chds.errBegin(&status)
        chds.datValid(self._locator, &valid, &status);
        raiseStarlinkException(status)

        return bool(valid)

    # Methods of the class.
    def annul(self):
        """Annuls the HDS locator"""
        cdef int status = chds.SAI__OK;
        #cdef HDSLOC* locpointer = self._locator
        chds.errBegin(&status);
        chds.datAnnul(&self._locator, &status);
        raiseStarlinkException(status)
        self._locator = NULL;

    def cell(self, indices):
        """returns new locator to a cell of an array.

        TODO: CALL SIGNATURE
        """
        cdef int status = chds.SAI__OK
        cdef int ndim = len(indices)
        cdef chds.HDSLoc * outloc = NULL

        # Convert from Python array to Fortran array: opposite order
        # for dimensions, and starting at 1.

        cdef chds.hdsdim rdim[chds.DAT__MXDIM]
        cdef int i
        for i in range(0, ndim):
            rdim[i] = indices[ndim -i -1] + 1

        chds.errBegin(&status);
        chds.datCell(self._locator, ndim, rdim, &outloc, &status)
        raiseStarlinkException(status)
        return HDSWrapperClass.from_pointer(outloc, owner=1)


    def find(self, compname):
        """
        Find a named component inside the HDS object.

        Returns an HDSWrapperObject about the new locator.
        Raises and HDSError if the component does not exist.

        args:
        -----
        compname: str, name of an existing component.

        """
        cdef int status = chds.SAI__OK
        cdef chds.HDSLoc * outloc = NULL
        compname = compname.encode('ascii')
        chds.errBegin(&status)
        chds.datFind(self._locator, compname, &outloc, &status)
        raiseStarlinkException(status)
        return HDSWrapperClass.from_pointer(outloc, owner=1)




    def get(self):
        """
        Get the data from the locator (regardless of type).
        Raises an error if the locator is not a type with data (e.g.a  struc).
        """
        cdef int state, status = chds.SAI__OK
        chds.errBegin(&status)

        # Check if its a structure!
        chds.datStruc(self._locator, &state, &status)
        raiseStarlinkException(status)
        if state == 1:
            raise StarlinkError("STRUCTURE ERROR: cannot use get on a structure")

        type_ = self.type
        shape = self.shape

        cdef cnp.npy_intp rdim[chds.DAT__MXDIM]
        cdef chds.hdsdim tdim[chds.DAT__MXDIM]
        cdef int ndim

        chds.datShape(self._locator, chds.DAT__MXDIM, tdim, &ndim, &status)
        raiseStarlinkException(status)

        cdef int i
        for i in range(0, ndim):
            rdim[i] = tdim[ndim -i -1]

        # Create a python array of the correct type.
        cdef int np_type =  _hdstype2numpy(type_.encode('ascii'))
        cdef cnp.ndarray mypyarray

        # If its a char* array, need to get the element size and pass
        # it in via PyArray_New. Otherwise use PyArray_SimpleNew.
        cdef int elsize
        useless = 0
        if type_[0:6] == '_CHAR*':
            elsize = self.clen
            mypyarray = cnp.PyArray_New(np.ndarray, ndim, rdim, np_type, NULL, NULL, elsize, 0, useless)
        else:
            mypyarray = cnp.PyArray_SimpleNew(ndim, rdim, np_type)

        # Get the actual data.
        chds.errBegin(&status)
        chds.datGet(self._locator, type_.encode('ascii'), ndim, tdim, mypyarray.data, &status)

        raiseStarlinkException(status)


        # if you are in _LOGICAL, need to convert back from INT to BOOL. Not sure why.
        if type_ == '_LOGICAL':
            mypyarray = mypyarray.astype(bool)
        return mypyarray


    def index(self, index):
        """ returns locator of index'th component (starts at 0)"""
        cdef int status = chds.SAI__OK
        cdef chds.HDSLoc* outloc = NULL
        chds.errBegin(&status);
        chds.datIndex(self._locator, index+1, &outloc, &status)
        raiseStarlinkException(status)
        return HDSWrapperClass.from_pointer(outloc, owner=1)


    def new(self, name, type, dims=None):
        """
        Create a new HDS structure benath the existing locator.

        Args:
        -----
        name: (str): name of new HDS object.
        type: (str) type of HDS object (e.g. NDF, _DOUBLE, ...)

        Optional Kwargs:
        ----------------
        dims: if given, implies scalar object.

        Returns
        -------
        Returns HDSWrapper object around locator for the new HDS object.
        """


        # Ensure we have bytes?
        name = name.encode('ascii')
        type = type.encode('ascii')

        cdef int status = chds.SAI__OK
        cdef chds.HDSLoc * outloc = NULL;
        cdef int i

        # Handl dimensions
        cdef int ndim=0
        cdef chds.hdsdim cdims[chds.DAT__MXDIM]

        chds.errBegin(&status)

        if dims:
            ndim = len(dims)
            for i in range(0, ndim):
                cdims[i] = dims[ndim -i -1]

        chds.datNew(self._locator, name, type, ndim, cdims, &status)
        chds.datFind(self._locator, name, &outloc, &status)

        raiseStarlinkException(status)
        return HDSWrapperClass.from_pointer(outloc, owner=1)


    def erase(self, name):
        """Recursively delete component name and its children"""
        cdef int status = chds.SAI__OK
        name = name.encode('ascii')
        chds.errBegin(&status)
        chds.datErase(self._locator, name, &status)
        raiseStarlinkException(status)


    def put(self, value):
        """
        Write a primitive inside an HDS object

        Note that this will force data to be converted to the type of
        the HDS object even if not safe. User beware.
        """
        cdef int status = chds.SAI__OK
        cdef char type_str[chds.DAT__SZTYP+1]

        # Get the type of the underlying structure.
        chds.errBegin(&status)
        chds.datType(self._locator, type_str, &status)
        raiseStarlinkException(status)


        # Create the array of the correct data type from the values.
        cdef chds.hdsdim hdims[chds.DAT__MXDIM];
        cdef int np_type = _hdstype2numpy(type_str)
        cdef int requirements = (cnp.NPY_ARRAY_DEFAULT | cnp.NPY_ARRAY_FORCECAST)

        cdef cnp.ndarray npyval
        npyval = cnp.PyArray_FROMANY(value, np_type, 0, chds.DAT__MXDIM, requirements )

        if npyval is None:
            raise StarlinkError('Could not create array for data')

        # Get the shape.
        cdef int ndim = npyval.ndim
        cdef int i
        npshape = npyval.shape
        for i in range(ndim):
            hdims[i] = npshape[ndim -i -1]

        # Call the HDS routine
        chds.errBegin(&status)
        chds.datPut(self._locator, type_str, ndim, hdims, &npyval.data[0], &status )
        raiseStarlinkException(status)



    # Wrapper
    @staticmethod
    cdef HDSWrapperClass from_pointer(chds.HDSLoc *_locator, bint owner=False):
        """Factory function:create HDSWrapperClass objects from
        given HDSLoc pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated."""


        # Check the status:
        #cdef const char * topic_str
        #cdef const char * extra
#        cdef int result
        cdef int status = chds.SAI__OK

        cdef HDSWrapperClass wrapper = HDSWrapperClass.__new__(HDSWrapperClass)
        wrapper._locator = _locator
        wrapper.ptr_owner = owner
        return wrapper

    def __repr__(self):
        cdef int status = chds.SAI__OK
        cdef int nlev = 0
        cdef char path[512]
        cdef char fname[512]
        if self._locator:

            chds.errBegin(&status)
            chds.hdsTrace(self._locator, &nlev, path, fname, &status, sizeof(path), sizeof(fname) )
            raiseStarlinkException(status)
            outstr = '<{}.{}>'.format(fname, path)
        else:
            outstr = "<DAT__NOLOC>"
        return outstr





