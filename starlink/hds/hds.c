//
// Python/C interface file for HDS files

/*
    Copyright 2009-2011 Tom Marsh
    Copyright 2011 Richard Hickman
    Copyright 2011 Tim Jenness
    Copyright 2018 East Asian Observatory
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

 */
//

#include <Python.h>
#include "structmember.h"
#include "numpy/arrayobject.h"

// Wrap the PyCObject -> PyCapsule transition to allow
// this to build with python2.
#include "npy_3kcompat.h"

#include <stdio.h>
#include <string.h>


// Starlink includes.
#include "hds.h"
#include "sae_par.h"


// Define the bad values: taken from prm_par.h
/* Bad values, used for flagging undefined data. */
#define VAL__BADUB 255
#define VAL__BADB (-127 - 1)
#define VAL__BADUW 65535
#define VAL__BADW (-32767 - 1)
#define VAL__BADI (-2147483647 - 1)
#define VAL__BADK (-9223372036854775807LL - 1)
#define VAL__BADR -3.40282347e+38F
#define VAL__BADD -1.7976931348623157e+308




static PyObject * StarlinkHDSError = NULL;

#if PY_VERSION_HEX >= 0x03000000
# define USE_PY3K
#endif

// Define an HDS object

typedef struct {
    PyObject_HEAD
    PyObject * _locator;
} HDSObject;

// Prototypes

static PyObject *
HDS_create_object( HDSLoc * loc );
static HDSLoc *
HDS_retrieve_locator( HDSObject * self );
static PyObject*
pydat_transfer(PyObject *self, PyObject *args);

static int
raiseHDSException( int *status );

// Deallocator. Annuls the locator and frees the object.

static void
HDS_dealloc(HDSObject * self)
{
    // Check that we didn't already annul the locator.
    if (self->_locator) {
        HDSLoc* loc = HDS_retrieve_locator(self);
        int status = SAI__OK;
        errBegin(&status);
        if (loc) datAnnul(&loc, &status);
        if (status != SAI__OK) errAnnul(&status);
        errEnd(&status);

        /* Frees the capsule object */
        Py_XDECREF(self->_locator);
    }

    PyObject_Del(self);
}

// Allocator of an HDS object

static PyObject *
HDS_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    HDSObject *self;

    self = (HDSObject *) _PyObject_New( type );
    if (self != NULL) {
      self->_locator = Py_None;
      if (self->_locator == NULL) {
        Py_DECREF(self);
        return NULL;
      }
    }

    return (PyObject *)self;
}

// __init__ method

static int
HDS_init(HDSObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *_locator = NULL;
    int result = -1;

    if ( PyArg_ParseTuple(args, "O", &_locator )) {
      result = 0;
      if (_locator) {
        PyObject * tmp = self->_locator;
        Py_INCREF(_locator);
        self->_locator = _locator;
        Py_XDECREF(tmp);
      }
    }
    return result;
}

// Extracts the contexts of the EMS error stack and raises an
// exception. Returns true if an exception was raised else
// false. Can be called as:
//   if (raiseHDSException(&status)) return NULL;
// The purpose of this routine is to flush errors and close
// the error context with an errEnd(). errBegin has be called
// in the code that is about to call Starlink routines.

#include "dat_err.h"
//#include "ndf_err.h"

static int
raiseHDSException( int *status )
{
  char param[ERR__SZPAR+1];
  char opstr[ERR__SZMSG+1];
  int parlen = 0;
  int oplen = 0;
  size_t stringlen = 1;
  PyObject * thisexc = NULL;
  char * errstring = NULL;

  if (*status == SAI__OK) {
    errEnd(status);
    return 0;
  }

  // We can translate some internal errors into standard python exceptions
  switch (*status) {
  case DAT__FILNF:
    thisexc = PyExc_IOError;
    break;
  default:
    thisexc = StarlinkHDSError;
  }

  // Start with a nul terminated buffer
  errstring = malloc( stringlen );
  if (!errstring) PyErr_NoMemory();
  errstring[0] = '\0';

  // Build up a string with the full error message
  while (*status != SAI__OK && errstring) {
    errLoad( param, sizeof(param), &parlen, opstr, sizeof(opstr), &oplen, status );
    if (*status != SAI__OK) {
      char *newstring;
      stringlen += oplen + 1;
      newstring = realloc( errstring, stringlen );
      if (newstring) {
        errstring = newstring;
        strcat( errstring, opstr );
        strcat( errstring, "\n" );
     } else {
        if (errstring) free(errstring);
        PyErr_NoMemory();
      }
    }
  }

  if (errstring) {
    PyErr_SetString( thisexc, errstring );
    free(errstring);
  }

  errEnd(status);
  return 1;
}

// Routine to convert an HDS type string to a numpy
// type code. Returns 0 and sets an exception
// if the HDS type is not recognized.

static int hdstype2numpy( const char * type ) {
  int retval = 0;

  if(strcmp(type,"_INTEGER") == 0) {
    retval = NPY_INT;
  } else if(strcmp(type,"_REAL") == 0) {
    retval = NPY_FLOAT;
  } else if(strcmp(type,"_DOUBLE") == 0) {
    retval = NPY_DOUBLE;
  } else if(strcmp(type,"_WORD") == 0) {
    retval = NPY_SHORT;
  } else if(strcmp(type,"_UWORD") == 0) {
    retval = NPY_USHORT;
  } else if(strcmp(type,"_BYTE") == 0) {
    retval = NPY_BYTE;
  } else if(strcmp(type,"_UBYTE") == 0) {
    retval = NPY_UBYTE;
  } else if(strcmp(type,"_LOGICAL") == 0) {
    retval = NPY_INT;
  } else if(strncmp(type,"_CHAR*",6) == 0) {
    retval = NPY_STRING;
  } else {
    // Set exception here
    PyErr_Format( PyExc_ValueError,
		  "Supplied HDS type '%s' does not correspond to a numpy type",
		  type);
  }
  return retval;
}

// Returns 1 if good, 0 if exception was thrown

static int numpy2hdsdim ( PyArrayObject *npyarr, int * ndim, hdsdim * hdims ) {

  int i = 0;

  *ndim = PyArray_NDIM( npyarr );
  if (*ndim > DAT__MXDIM) {
    PyErr_Format( PyExc_ValueError,
		  "Supplied numpy array has more than %d dimensions",
		  DAT__MXDIM );
    return 0;
  }

  for (i=0; i<*ndim;i++) {
    hdims[i] = PyArray_DIM(npyarr, *ndim - i - 1);
  }
  return 1;
}

// Now onto main routines

// Annuls the locator but does not free the object

static PyObject*
pydat_annul(HDSObject *self)
{
    // Recover C-pointer passed via Python
    HDSLoc* loc = HDS_retrieve_locator(self);
    int status = SAI__OK;
    errBegin(&status);
    datAnnul(&loc, &status);

    /* Free the capsule object before loosing the pointer to it. */
    Py_XDECREF(self->_locator);
    self->_locator = NULL;

    if(raiseHDSException(&status)) return NULL;
    Py_RETURN_NONE;
};

static PyObject*
pydat_cell(HDSObject *self, PyObject *args)
{
    PyObject *pobj1, *osub;
    if(!PyArg_ParseTuple(args, "O:pydat_cell", &osub))
	return NULL;

    // Recover C-pointer passed via Python
    HDSLoc* loc1 = HDS_retrieve_locator(self);

    // Attempt to convert the input to something useable
    PyArrayObject *sub = (PyArrayObject *) PyArray_ContiguousFromAny(osub, NPY_INT, 1, 1);
    if(!sub) return NULL;

    // Convert Python-like --> Fortran-like
    int ndim = PyArray_SIZE(sub);
    int i, rdim[ndim];
    int *sdata = (int*)PyArray_DATA(sub);
    for(i=0; i<ndim; i++) rdim[i] = sdata[ndim-i-1]+1;

    HDSLoc* loc2 = NULL;
    int status = SAI__OK;
    errBegin(&status);
    // Finally run the routine
    datCell(loc1, ndim, rdim, &loc2, &status);
    if(status != SAI__OK) goto fail;
    errEnd(&status);

    // PyCObject to pass pointer along to other wrappers
    Py_DECREF(sub);
    return HDS_create_object(loc2);

fail:
    raiseHDSException(&status);
    Py_XDECREF(sub);
    return NULL;
};

static PyObject*
pydat_find(HDSObject *self, PyObject *args)
{
    PyObject* pobj1;
    const char* name;
    if(!PyArg_ParseTuple(args, "s:pydat_find", &name))
	return NULL;

    // Recover C-pointer passed via Python
    HDSLoc* loc1 = HDS_retrieve_locator( self );
    HDSLoc* loc2 = NULL;

    int status = SAI__OK;
    errBegin(&status);
    datFind(loc1, name, &loc2, &status);
    if (raiseHDSException(&status)) return NULL;

    // PyCObject to pass pointer along to other wrappers
    return HDS_create_object(loc2);
};

static PyObject*
pydat_get(HDSObject *self)
{
    // Recover C-pointer passed via Python
    HDSLoc* loc = HDS_retrieve_locator(self);

    // guard against structures
    int state, status = SAI__OK;
    errBegin(&status);
    datStruc(loc, &state, &status);
    if (raiseHDSException(&status)) return NULL;
    if(state){
	PyErr_SetString(PyExc_IOError, "dat_get error: cannot use on structures");
	return NULL;
    }

    // get type
    char typ_str[DAT__SZTYP+1];
    errBegin(&status);
    datType(loc, typ_str, &status);

    // get shape
    const int NDIMX=7;
    int ndim;
    hdsdim tdim[NDIMX];
    datShape(loc, NDIMX, tdim, &ndim, &status);
    if (raiseHDSException(&status)) return NULL;

    PyArrayObject* arr = NULL;

    // Either return values as a single scalar or a numpy array

    // Reverse order of dimensions
    npy_intp rdim[NDIMX];
    int i;
    for(i=0; i<ndim; i++) rdim[i] = tdim[ndim-i-1];
    errBegin(&status);

    if(strcmp(typ_str, "_INTEGER") == 0){
        arr = (PyArrayObject*) PyArray_SimpleNew(ndim, rdim, NPY_INT);
    }else if(strcmp(typ_str, "_LOGICAL") == 0){
        arr = (PyArrayObject*) PyArray_SimpleNew(ndim, rdim, NPY_INT);
    }else if(strcmp(typ_str, "_REAL") == 0){
	arr = (PyArrayObject*) PyArray_SimpleNew(ndim, rdim, NPY_FLOAT);
    }else if(strcmp(typ_str, "_DOUBLE") == 0){
	arr = (PyArrayObject*) PyArray_SimpleNew(ndim, rdim, NPY_DOUBLE);
    }else if(strncmp(typ_str, "_CHAR", 5) == 0){

	// work out the number of bytes
	size_t nbytes;
	datLen(loc, &nbytes, &status);
	if (status != SAI__OK) goto fail;

	int ncdim = 1+ndim;
	int cdim[ncdim];
	cdim[0] = nbytes+1;
	for(i=0; i<ndim; i++) cdim[i+1] = rdim[i];

	PyArray_Descr *descr = PyArray_DescrNewFromType(NPY_STRING);
	descr->elsize = nbytes;
	arr = (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type, descr, ndim, rdim,
						    NULL, NULL, 0, NULL);

    }else if(strcmp(typ_str, "_WORD") == 0){
	arr = (PyArrayObject*) PyArray_SimpleNew(ndim, rdim, NPY_SHORT);
    }else if(strcmp(typ_str, "_UWORD") == 0){
	arr = (PyArrayObject*) PyArray_SimpleNew(ndim, rdim, NPY_USHORT);
    }else if(strcmp(typ_str, "_BYTE") == 0){
	arr = (PyArrayObject*) PyArray_SimpleNew(ndim, rdim, NPY_BYTE);
    }else if(strcmp(typ_str, "_UBYTE") == 0){
	arr = (PyArrayObject*) PyArray_SimpleNew(ndim, rdim, NPY_UBYTE);
    }else{
	PyErr_SetString(PyExc_IOError, "dat_get: encountered an unimplemented type");
	return NULL;
    }
    if(arr == NULL) goto fail;
    datGet(loc, typ_str, ndim, tdim, arr->data, &status);
    if(status != SAI__OK) goto fail;
    errEnd(&status);

    if (strcmp(typ_str, "_LOGICAL") == 0) {
        // convert back to Boolean.
        PyArray_Descr* typedescr = PyArray_DescrFromType(NPY_BOOL);
        arr = PyArray_CastToType(arr, typedescr, 0);
      }
    return PyArray_Return(arr);

fail:
    raiseHDSException(&status);
    Py_XDECREF(arr);
    return NULL;

};

static PyObject*
pydat_index(HDSObject *self, PyObject *args)
{
    PyObject* pobj;
    int index;
    if(!PyArg_ParseTuple(args, "i:pydat_index", &index))
	return NULL;

    // Recover C-pointer passed via Python
    HDSLoc* loc1 = HDS_retrieve_locator(self);
    HDSLoc* loc2 = NULL;

    int status = SAI__OK;
    errBegin(&status);
    datIndex(loc1, index+1, &loc2, &status);
    if(raiseHDSException(&status)) return NULL;
    return HDS_create_object(loc2);
};

// make a new HDS file or make a new HDS structure.
// Choice depends on whether "self" refers to a locator or not
// Always returns the locator to the created structure
// This is not how datNew normally behaves but is how hdsNew works.
static PyObject*
pydat_new(HDSObject *self, PyObject *args)
{
        const char *type, *name, *file;
	int ndim = 0;
	PyArrayObject * dims = NULL;
	PyObject *dims_object = NULL;
	HDSLoc* loc = HDS_retrieve_locator(self);
	HDSLoc* outloc = NULL;
	hdsdim hdims[DAT__MXDIM];
	int status = SAI__OK;

	if (!loc) {
	  // We are creating a new HDS file
	  // Optional dims implies scalar
	  if (!PyArg_ParseTuple( args, "sss|O:pyhds_new",
				 &file, &name, &type, &dims_object ))
	    return NULL;
	} else {
	  // Creating HDS component
	  if(!PyArg_ParseTuple(args, "ss|O:pydat_new", &name, &type, &dims_object))
	    return NULL;
	}

	if (dims_object) {
	  /* Note that HDS dimensions are hdsdim type so we would need to copy
	     or work out what PyArray type to use */
	  dims = (PyArrayObject *)PyArray_ContiguousFromAny( dims_object,
							     PyArray_INT, 0,1);
	  if (dims) {
	    int i;
	    int *npydims = PyArray_DATA(dims);
	    ndim = PyArray_Size(dims);
	    for (i = 0; i< ndim; i++) {
	      hdims[i] = npydims[ndim - i - 1];
	    }
	  }
	}

	errBegin(&status );

	if (!loc) {
	  hdsNew( file, name, type, ndim, (dims ? hdims : NULL),
		  &outloc, &status );
	} else {
	  // We are creating an HDS component
	  datNew( loc, name, type,ndim, (dims ?  hdims : NULL),
		  &status);
	  datFind( loc, name, &outloc, &status );
	}
	Py_XDECREF( dims );
	if (raiseHDSException(&status))
		return NULL;
	return HDS_create_object(outloc);
}

// open an HDS file
static PyObject *
pyhds_open( HDSObject *self, PyObject *args )
{
  const char * file = NULL;
  const char * mode = NULL;

  if(!PyArg_ParseTuple(args,"ss:pyhds_open",&file, &mode))
    return NULL;

  int status = SAI__OK;
  HDSLoc * loc = NULL;
  errBegin(&status);

  hdsOpen( file, mode, &loc, &status );
  if (raiseHDSException(&status))
    return NULL;
  return HDS_create_object(loc);
}

// write a primitive
static PyObject*
pydat_put(HDSObject *self, PyObject *args)
{
	PyObject *value, *dimobj;
	PyArrayObject *npyval;
	char type[DAT__SZTYP+1];
	PyArrayObject * dims = NULL;
	PyObject *dims_object = NULL;
	hdsdim hdims[DAT__MXDIM];
	int ndim;
        int requirements;

	if(!PyArg_ParseTuple(args,"O:pydat_put",&value))
		return NULL;
	HDSLoc* loc = HDS_retrieve_locator(self);

	// Work out the type of the underlying HDS structure
	int status = SAI__OK;
        errBegin(&status);
	datType( loc, type, &status );
	if (raiseHDSException(&status)) return NULL;

	// create a pointer to an array of the appropriate data type
	int npytype = hdstype2numpy( type );


        // HDS behaviour is to always force the input data into the
        // requested format. Therefore set the PyArray flag
        // NPY_ARRAY_FORCECAST so that it will convert even if it is not
        // safe. Requires using PyArray_FromAny instead of
        // PyArray_ContiguousFromAny.
        requirements = (NPY_ARRAY_DEFAULT | NPY_ARRAY_FORCECAST);
        npyval = (PyArrayObject*) PyArray_FROMANY( value, npytype, 0, DAT__MXDIM, requirements );
        if ( !npyval ) return NULL;


	void *valptr = PyArray_DATA(npyval);
	if (!numpy2hdsdim( npyval, &ndim, hdims )) return NULL;
	errBegin(&status);
	datPut( loc, type, ndim, hdims, valptr, &status );
	Py_XDECREF(npyval);

	if (raiseHDSException(&status))
		return NULL;
	Py_RETURN_NONE;
}

static PyObject*
pydat_putc(HDSObject *self, PyObject *args)
{
	PyObject *strobj,*locobj;
	int strlen;
	if(!PyArg_ParseTuple(args,"Oi:pydat_putc",&strobj,&strlen))
		return NULL;
	HDSLoc *loc = HDS_retrieve_locator(self);
	PyArrayObject *npystr = (PyArrayObject*) PyArray_FROM_OTF(strobj,NPY_STRING,NPY_FORCECAST);
	char *strptr = PyArray_DATA(npystr);
	int status = SAI__OK;
        errBegin(&status);
	datPutC(loc,0,0,strptr,(size_t)strlen,&status);
	if (raiseHDSException(&status))
		return NULL;
	Py_DECREF(npystr);
	Py_RETURN_NONE;
}

//
// METHODS THAT IMPLEMENT ATTRIBUTES
//

static PyObject*
pydat_clen(HDSObject *self)
{
    // Recover C-pointer passed via Python
    HDSLoc* loc = HDS_retrieve_locator(self);

    int status = SAI__OK;
    size_t clen;
    errBegin(&status);
    datClen(loc, &clen, &status);
    if (raiseHDSException(&status)) return NULL;

    return Py_BuildValue("i", clen);
};

static PyObject*
pydat_name(HDSObject *self)
{
    // Recover C-pointer passed via Python
    HDSLoc* loc = HDS_retrieve_locator(self);

    char name_str[DAT__SZNAM+1];
    int status = SAI__OK;
    errBegin(&status);
    datName(loc, name_str, &status);
    if (raiseHDSException(&status)) return NULL;
    return Py_BuildValue("s", name_str);
};

static PyObject*
pydat_ncomp(HDSObject *self)
{
    // Recover C-pointer passed via Python
    HDSLoc* loc = HDS_retrieve_locator(self);

    int status = SAI__OK, ncomp;
    errBegin(&status);
    datNcomp(loc, &ncomp, &status);
    if (raiseHDSException(&status)) return NULL;

    return Py_BuildValue("i", ncomp);
};

static PyObject*
pydat_shape(HDSObject *self)
{
    // Recover C-pointer passed via Python
    HDSLoc* loc = HDS_retrieve_locator(self);

    const int NDIMX=7;
    int ndim;
    hdsdim tdim[NDIMX];
    int status = SAI__OK;
    errBegin(&status);
    datShape(loc, NDIMX, tdim, &ndim, &status);
    if (raiseHDSException(&status)) return NULL;

    // Return None in this case
    if(ndim == 0) Py_RETURN_NONE;

    // Create array of correct dimensions to save data to
    PyArrayObject* dim = NULL;
    npy_intp sdim[1];
    int i;
    sdim[0] = ndim;
    dim = (PyArrayObject*) PyArray_SimpleNew(1, sdim, PyArray_INT);
    if(dim == NULL) goto fail;

    // Reverse order Fortran --> C convention
    int* sdata = (int*)dim->data;
    for(i=0; i<ndim; i++) sdata[i] = tdim[ndim-i-1];
    return Py_BuildValue("N", PyArray_Return(dim));

fail:
    Py_XDECREF(dim);
    return NULL;
};


static PyObject*
pydat_state(HDSObject *self)
{
    // Recover C-pointer passed via Python
    HDSLoc* loc = HDS_retrieve_locator(self);

    int status = SAI__OK, state;
    errBegin(&status);
    datState(loc, &state, &status);
    if (raiseHDSException(&status)) return NULL;
    return PyBool_FromLong( state );
};

static PyObject*
pydat_struc(HDSObject *self)
{
    // Recover C-pointer passed via Python
    HDSLoc* loc = HDS_retrieve_locator(self);

    // guard against structures
    int state, status = SAI__OK;
    errBegin(&status);
    datStruc(loc, &state, &status);
    if (raiseHDSException(&status)) return NULL;
    return PyBool_FromLong( state );
};

static PyObject*
pydat_type(HDSObject *self)
{
    // Recover C-pointer passed via Python
    HDSLoc* loc = HDS_retrieve_locator(self);

    char typ_str[DAT__SZTYP+1];
    int status = SAI__OK;
    errBegin(&status);
    datType(loc, typ_str, &status);
    if (raiseHDSException(&status)) return NULL;
    return Py_BuildValue("s", typ_str);
};

static PyObject*
pydat_valid(HDSObject *self)
{
    // Recover C-pointer passed via Python
    HDSLoc* loc = HDS_retrieve_locator(self);

    int state, status = SAI__OK;
    errBegin(&status);
    datValid(loc, &state, &status);
    if (raiseHDSException(&status)) return NULL;

    return PyBool_FromLong( state );
};


//
//
//  END OF METHODS - NOW DEFINE ATTRIBUTES AND MODULES

static PyObject *HDS_repr( PyObject * self ) {
        char buff[1024];  /* To receive represenation */
        char path[512];
        char fname[512];
        PyObject *result = NULL;
        HDSLoc *loc = HDS_retrieve_locator((HDSObject*)self);

        if (loc) {
          int nlev = 0;
          int status = SAI__OK;
          hdsTrace( loc, &nlev, path, fname, &status, sizeof(path), sizeof(fname) );
          snprintf( buff, sizeof(buff), "<%s.%s>",
                    fname,path);
        } else {
          snprintf( buff, sizeof(buff), "%s", "<DAT__NOLOC>");
        }
        result = Py_BuildValue( "s", buff );

        return result;
}

static PyMemberDef HDS_members[] = {
  {"_locator", T_OBJECT_EX, offsetof(HDSObject, _locator), 0,
   "HDS Locator"},
  {NULL} /* Sentinel */
};

// Accessor methods - all are readonly

static PyGetSetDef HDS_getseters[] = {
  { "clen", (getter)pydat_clen, NULL, "Character string length of primitive object", NULL },
  { "name", (getter)pydat_name, NULL, "HDS component name", NULL },
  { "ncomp", (getter)pydat_ncomp, NULL, "Number of components in structure", NULL},
  { "shape", (getter)pydat_shape, NULL, "Shape of component (None for a scalar)", NULL },
  { "state", (getter)pydat_state, NULL, "The state of the HDS component", NULL},
  { "struc", (getter)pydat_struc, NULL, "Is the component a structure?", NULL},
  { "type", (getter)pydat_type, NULL, "Type of the HDS component", NULL},
  { "valid", (getter)pydat_valid, NULL, "Is the locator valid?", NULL},
  {NULL} /* Sentinel */
};

static PyObject*
pyhds_getbadvalue(HDSObject *self, PyObject *args)
{
	const char *type;
	if(!PyArg_ParseTuple(args, "s:pyhds_getbadvalue", &type))
		return NULL;
	if (strcmp(type,"_DOUBLE") == 0)
		return Py_BuildValue("f",VAL__BADD);
	else if (strcmp(type,"_REAL") == 0)
		return Py_BuildValue("f",VAL__BADR);
	else if (strcmp(type,"_INTEGER") == 0)
		return Py_BuildValue("i",VAL__BADI);
        else if (strcmp(type, "_BYTE") == 0)
                return Py_BuildValue("i",VAL__BADB);
        else if (strcmp(type, "_UBYTE") == 0)
                return Py_BuildValue("i", VAL__BADUB);
        else if (strcmp(type, "_WORD") == 0)
                return Py_BuildValue("i", VAL__BADW);
        else if (strcmp(type, "_UWORD") == 0)
                return Py_BuildValue("i", VAL__BADUW);
        //        else if (strcmp(type, "_INT64") == 0)
        // return Py_BuildValue("i", VAL__BADK);
	else
          PyErr_Format(PyExc_ValueError, "type must be one of _DOUBLE, _REAL, _INTEGER, _BYTE, _UBYTE, _WORD, or U_WORD");
		return NULL;
}

// module methods
static PyMethodDef HDS_module_methods[] = {

  {"_transfer", (PyCFunction)pydat_transfer, METH_VARARGS,
   "starlink.hds.api.transfer(xloc) -- transfer HDS locator from NDF."},

  {"new", (PyCFunction)pydat_new, METH_VARARGS,
   "loc = new(filename, hdsname, type, dims) -- create a new HDS structure and return a locator. Dims is optional and implies a scalar object."},

  {"open", (PyCFunction)pyhds_open, METH_VARARGS,
   "loc = open(name, mode) -- open an existing HDS file with mode 'READ', 'WRITE' or 'UPDATE'"},

  {"getbadvalue", (PyCFunction)pyhds_getbadvalue, METH_VARARGS,
   "getbadvalue(type) -- return the bad pixel value for given HDS numerical data type."},
  {NULL, NULL, 0, NULL} /* Sentinel */
};

// The methods of an hdsloc object
static PyMethodDef HDSloc_methods[] = {

  {"_transfer", (PyCFunction)pydat_transfer, METH_VARARGS,
   "starlink.hds.api.transfer(xloc) -- transfer HDS locator from NDF."},

  {"annul", (PyCFunction)pydat_annul, METH_NOARGS,
   "hdsloc.annul() -- annuls the HDS locator."},

  {"cell", (PyCFunction)pydat_cell, METH_VARARGS,
   "loc2 = hdsloc1.cell(sub) -- returns locator of a cell of an array."},

  {"find", (PyCFunction)pydat_find, METH_VARARGS,
   "loc2 = hdsloc1.find(name) -- finds a named component, returns locator."},

  {"get", (PyCFunction)pydat_get, METH_NOARGS,
   "value = hdsloc.get() -- get data associated with locator regardless of type."},

  {"index", (PyCFunction)pydat_index, METH_VARARGS,
   "loc2 = hdsloc1.index(index) -- returns locator of index'th component (starts at 0)."},

  {"new", (PyCFunction)pydat_new, METH_VARARGS,
   "newloc = hdsloc.new(name,type,dims) -- create a new HDS structure beneath the existing locator. Dims is optional and implies a scalar object."},

  {"put", (PyCFunction)pydat_put, METH_VARARGS,
   "status = hdsloc.put(value) -- write a primitive inside an hds item."},

  {"putc", (PyCFunction)pydat_putc, METH_VARARGS,
   "hdsloc.putc(string) -- write a character string to primitive at locator."},

  {NULL, NULL, 0, NULL} /* Sentinel */
};



static PyTypeObject HDSType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "starlink.hds",                /* tp_name */
    sizeof(HDSObject),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)HDS_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    HDS_repr,                  /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "HDS Locator type: Raw API for HDS access",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    HDSloc_methods,             /* tp_methods */
    HDS_members,             /* tp_members */
    HDS_getseters,             /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)HDS_init,      /* tp_init */
    0,                         /* tp_alloc */
    HDS_new,                 /* tp_new */
};

// Helper to create an object with an HDS locator

static PyObject *
HDS_create_object( HDSLoc * locator )
{
  HDSObject * self = (HDSObject*)HDS_new( &HDSType, NULL, NULL );
  self->_locator = NpyCapsule_FromVoidPtr( locator, NULL );
  return Py_BuildValue("N", self);
}

static HDSLoc *
HDS_retrieve_locator( HDSObject *self)
{
  if (self) {
    return (HDSLoc*)NpyCapsule_AsVoidPtr(self->_locator);
  } else {
    return NULL;
  }
}

static PyObject*
pydat_transfer(PyObject *self, PyObject *args)
{
  HDSObject * newself = (HDSObject*)HDS_new( &HDSType, NULL, NULL );
  if (!newself) return NULL;
  HDS_init( newself, args, NULL);
  return (PyObject*)newself;
}

#ifdef USE_PY3K

#define RETVAL m

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "hds",
  "Raw HDS API",
  -1,
  HDS_module_methods,
  NULL,
  NULL,
  NULL,
  NULL
};

PyObject *PyInit_hds(void)
#else

#define RETVAL

PyMODINIT_FUNC
inithds(void)
#endif
{
    PyObject *m = NULL;

    if (PyType_Ready(&HDSType) < 0)
        return RETVAL;

#ifdef USE_PY3K
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("hds", HDS_module_methods,
                      "Raw HDS API:");
#endif
    import_array();

    Py_INCREF(&HDSType);
    PyModule_AddObject(m, "hds", (PyObject *)&HDSType);

    StarlinkHDSError = PyErr_NewException("starlink.hds.error", NULL, NULL);
    Py_INCREF(StarlinkHDSError);
    PyModule_AddObject(m, "error", StarlinkHDSError);

    return RETVAL;
}
