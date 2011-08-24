//
// Python/C interface file for ndf files
//
// This implements a class such that the NDF identifier is used
// as an object.

/*
    Copyright 2009-2011 Tom Marsh
    Copyright 2011 Richard Hickman
    Copyright 2011 Tim Jenness
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

// NDF includes
#include "ndf.h"
#include "mers.h"
#include "star/hds.h"
#include "sae_par.h"
#include "prm_par.h"

static PyObject * StarlinkNDFError = NULL;

#if PY_VERSION_HEX >= 0x03000000
# define USE_PY3K
#endif

// Define a NDF object

typedef struct {
    PyObject_HEAD
    int _ndfid;
    int _place;
} NDF;

// Prototypes

static PyObject *
NDF_create_object( int ndfid, int place );

// Deallocator of this object
// - we do annul the NDF identifier

static void
NDF_dealloc(NDF* self)
{
    int status = SAI__OK;
    errBegin(&status);
    if (self->_ndfid != NDF__NOID) ndfAnnul( &self->_ndfid, &status);
    if (status != SAI__OK) errAnnul(&status);
    errEnd(&status);
    PyObject_Del( self );
}

// Allocator of an NDF object

static PyObject *
NDF_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    NDF *self;

    self = (NDF *) _PyObject_New( type );
    if (self != NULL) {
        self->_ndfid = NDF__NOID;
        self->_place = NDF__NOPL;
    }

    return (PyObject *)self;
}

// __init__ method

static int
NDF_init(NDF *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"_ndfid", "_place", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist,
                                      &self->_ndfid, &self->_place ))
        return -1;

    return 0;
}

// Removes locators once they are no longer needed

static void PyDelLoc_ptr(void *ptr)
{
    HDSLoc* loc = (HDSLoc*)ptr;
    int status = SAI__OK;
    datAnnul(&loc, &status);
    printf("Inside PyDelLoc\n");
    return;
}

// Need a PyCapsule version for Python3

#ifdef USE_PY3K
static void PyDelLoc( PyObject *cap )
{
  PyDelLoc_ptr( PyCapsule_GetPointer( cap, NULL ));
}
#else
static void PyDelLoc( void * ptr )
{
  PyDelLoc_ptr(ptr);
}
#endif

// Translates from Python's axis number into the one NDF expects
// Needed because of inversion of C and Fortran arrays.
// Requires knowing the number of dimensions.

static int tr_iaxis(int indf, int iaxis, int *status)
{
    if(iaxis == -1) return 0;

    // Get dimensions
    const int NDIMX = 10;
    int ndim, idim[NDIMX];
    ndfDim(indf, NDIMX, idim, &ndim, status);
    if(*status != SAI__OK) return -1;
    if(iaxis < -1 || iaxis > ndim-1){
	PyErr_SetString(PyExc_IOError, "tr_axis: axis number too out of range");
	*status = SAI__ERROR;
	return -1;
    }
    return ndim-iaxis;
}

// Extracts the contexts of the EMS error stack and raises an
// exception. Returns true if an exception was raised else
// false. Can be called as:
//   if (raiseNDFException(&status)) return NULL;
// The purpose of this routine is to flush errors and close
// the error context with an errEnd(). errBegin has be called
// in the code that is about to call Starlink routines.

#include "dat_err.h"
#include "ndf_err.h"

static int
raiseNDFException( int *status )
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
    thisexc = StarlinkNDFError;
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

// Now onto main routines

static PyObject*
pyndf_acget(NDF *self, PyObject *args)
{
    const char *comp;
    int iaxis;
    if(!PyArg_ParseTuple(args, "si:pyndf_acget", &comp, &iaxis))
	return NULL;

    // Return None if component does not exist
    int state, status = SAI__OK;
    int naxis = tr_iaxis(self->_ndfid, iaxis, &status);
    errBegin(&status);
    ndfAstat(self->_ndfid, comp, naxis, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    if(!state)
	Py_RETURN_NONE;

    int clen;
    ndfAclen(self->_ndfid, comp, naxis, &clen, &status);
    char value[clen+1];
    ndfAcget(self->_ndfid, comp, naxis, value, clen+1, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("s", value);
};

static PyObject*
pyndf_aform(NDF *self, PyObject *args)
{
    const char *comp;
    int iaxis;
    if(!PyArg_ParseTuple(args, "si:pyndf_aform", &comp, &iaxis))
	return NULL;
    int status = SAI__OK;
    int naxis = tr_iaxis(self->_ndfid, iaxis, &status);
    char value[30];
    errBegin(&status);
    ndfAform(self->_ndfid, comp, naxis, value, 30, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("s", value);
};

// THINK - THIS IS A DESTRUCTOR
static PyObject*
pyndf_annul(NDF *self)
{
    int status = SAI__OK;
    errBegin(&status);
    ndfAnnul(&self->_ndfid, &status);
    if (raiseNDFException(&status)) return NULL;
    Py_RETURN_NONE;
};

static PyObject*
pyndf_anorm(NDF *self, PyObject *args)
{
    int iaxis;
    if(!PyArg_ParseTuple(args, "i:pyndf_anorm", &iaxis))
	return NULL;
    int state, status = SAI__OK;
    errBegin(&status);
    int naxis = tr_iaxis(self->_ndfid, iaxis, &status);
    ndfAnorm(self->_ndfid, naxis, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("i", state);
};

static PyObject*
pyndf_aread(NDF *self, PyObject *args)
{
    int iaxis;
    const char *MMOD = "READ";
    const char *comp;
    if(!PyArg_ParseTuple(args, "si:pyndf_aread", &comp, &iaxis))
	return NULL;

    int status = SAI__OK;
    errBegin(&status);
    int naxis = tr_iaxis(self->_ndfid, iaxis, &status);

    // Return None if component does not exist
    int state;
    ndfAstat(self->_ndfid, comp, naxis, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    if(!state) Py_RETURN_NONE;

    // Get dimensions
    const int NDIMX = 10;
    int idim[NDIMX], ndim;
    ndfDim(self->_ndfid, NDIMX, idim, &ndim, &status);
    if (raiseNDFException(&status)) return NULL;

    // get number for particular axis in question.
    int nelem = idim[naxis-1];

    // Determine the data type
    const int MXLEN=33;
    char type[MXLEN];
    ndfAtype(self->_ndfid, comp, naxis, type, MXLEN, &status);
    if (raiseNDFException(&status)) return NULL;

    // Create array of correct dimensions and type to save data to
    size_t nbyte;
    ndim = 1;
    npy_intp dim[1] = {nelem};
    PyArrayObject* arr = NULL;
    if(strcmp(type, "_REAL") == 0){
	arr = (PyArrayObject*) PyArray_SimpleNew(ndim, dim, PyArray_FLOAT);
	nbyte = sizeof(float);
    }else if(strcmp(type, "_DOUBLE") == 0){
	arr = (PyArrayObject*) PyArray_SimpleNew(ndim, dim, PyArray_DOUBLE);
	nbyte = sizeof(double);
    }else if(strcmp(type, "_INTEGER") == 0){
	arr = (PyArrayObject*) PyArray_SimpleNew(ndim, dim, PyArray_INT);
	nbyte = sizeof(int);
    }else{
	PyErr_SetString(PyExc_IOError, "ndf_aread error: unrecognised data type");
	goto fail;
    }
    if(arr == NULL) goto fail;

    // map, store, unmap
    int nread;
    void *pntr[1];
    ndfAmap(self->_ndfid, comp, naxis, type, MMOD, pntr, &nread, &status);
    if (status != SAI__OK) goto fail;
    if(nelem != nread){
	printf("nread = %d, nelem = %d, iaxis = %d, %d\n",nread,nelem,iaxis,naxis);
	PyErr_SetString(PyExc_IOError, "ndf_aread error: number of elements different from number expected");
	goto fail;
    }
    memcpy(arr->data, pntr[0], nelem*nbyte);
    ndfAunmp(self->_ndfid, comp, naxis, &status);

    return Py_BuildValue("N", PyArray_Return(arr));

fail:
    raiseNDFException(&status);
    Py_XDECREF(arr);
    return NULL;

};

static PyObject*
pyndf_astat(NDF *self, PyObject *args)
{
    const char *comp;
    int iaxis;
    if(!PyArg_ParseTuple(args, "si:pyndf_astat", &comp, &iaxis))
	return NULL;
    int state, status = SAI__OK;
    errBegin(&status);
    int naxis = tr_iaxis(self->_ndfid, iaxis, &status);

    ndfAstat(self->_ndfid, comp, naxis, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("i", state);
};

static PyObject*
pyndf_init(NDF *self, PyObject *args)
{
    int argc = 0, status = SAI__OK;
    char **argv = NULL;
    errBegin(&status);
    ndfInit(argc, argv, &status);
    if (raiseNDFException(&status)) return NULL;
    Py_RETURN_NONE;
};

static PyObject*
pyndf_begin(NDF *self)
{
    ndfBegin();
    Py_RETURN_NONE;
};

static PyObject*
pyndf_bound(NDF *self)
{
    int i;

    PyArrayObject* bound = NULL;
    int ndim;
    const int NDIMX=20;
    int *lbnd = malloc(NDIMX*sizeof(int));
    int *ubnd = malloc(NDIMX*sizeof(int));
    if(lbnd == NULL || ubnd == NULL)
	goto fail;

    int status = SAI__OK;
    errBegin(&status);
    ndfBound(self->_ndfid, NDIMX, lbnd, ubnd, &ndim, &status );
    if(status != SAI__OK) goto fail;

    npy_intp odim[2];
    odim[0] = 2;
    odim[1] = ndim;
    bound   = (PyArrayObject*) PyArray_SimpleNew(2, odim, PyArray_INT);
    if(bound == NULL) goto fail;
    int *bptr = (int *)bound->data;
    for(i=0; i<ndim; i++){
	bptr[i]      = lbnd[ndim-i-1];
	bptr[i+ndim] = ubnd[ndim-i-1];
    }
    free(lbnd);
    free(ubnd);
    return Py_BuildValue("N", PyArray_Return(bound));

fail:
    raiseNDFException(&status);
    if(lbnd != NULL) free(lbnd);
    if(ubnd != NULL) free(ubnd);
    Py_XDECREF(bound);
    return NULL;
};

static PyObject*
pyndf_cget(NDF *self, PyObject *args)
{
    const char *comp;
    if(!PyArg_ParseTuple(args, "s:pyndf_cget", &comp))
	return NULL;

    // Return None if component does not exist
    int state, status = SAI__OK;
    errBegin(&status);
    ndfState(self->_ndfid, comp, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    if(!state)
	Py_RETURN_NONE;

    int clen;
    errBegin(&status);
    ndfClen(self->_ndfid, comp, &clen, &status);
    if (raiseNDFException(&status)) return NULL;
    char value[clen+1];
    errBegin(&status);
    ndfCget(self->_ndfid, comp, value, clen+1, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("s", value);
};

static PyObject*
pyndf_dim(NDF *self)
{
    int i;

    PyArrayObject* dim = NULL;
    int ndim;
    const int NDIMX=20;
    int *idim = (int *)malloc(NDIMX*sizeof(int));
    if(idim == NULL)
	goto fail;

    int status = SAI__OK;
    errBegin(&status);
    ndfDim(self->_ndfid, NDIMX, idim, &ndim, &status );
    if(status != SAI__OK) goto fail;

    npy_intp odim[1];
    odim[0] = ndim;
    dim = (PyArrayObject*) PyArray_SimpleNew(1, odim, PyArray_INT);
    if(dim == NULL) goto fail;
    for(i=0; i<ndim; i++)
	((int *)dim->data)[i] = idim[ndim-i-1];
    free(idim);

    return Py_BuildValue("N", PyArray_Return(dim));

fail:
    raiseNDFException(&status);
    if(idim != NULL) free(idim);
    Py_XDECREF(dim);
    return NULL;
};

static PyObject*
pyndf_end(NDF *self)
{
    int status = SAI__OK;
    errBegin(&status);
    ndfEnd(&status);
    if (raiseNDFException(&status)) return NULL;
    Py_RETURN_NONE;
};

// open an existing or new NDF file
static PyObject*
pyndf_open(NDF *self, PyObject *args)
{
    const char *name;
	const char *mode = "READ";
	const char *stat = "OLD";
    if(!PyArg_ParseTuple(args, "s|ss:pyndf_open", &name, &mode, &stat))
		return NULL;
	// check for allowed values of mode and stat
    if(strcmp(mode,"READ") != 0 && strcmp(mode,"WRITE") != 0 && strcmp(mode,"UPDATE") != 0) {
        PyErr_SetString( PyExc_ValueError, "Incorrect mode for ndf_open");
        return NULL;
    }
    if(strcmp(stat,"OLD") != 0 && strcmp(stat,"NEW") != 0 && strcmp(stat,"UNKNOWN") != 0) {
        PyErr_SetString( PyExc_ValueError, "Unknown status string for ndf_open" );
        return NULL;
    }
    int indf, place;
    int status = SAI__OK;
    errBegin(&status);
    indf = NDF__NOID;
    place = NDF__NOPL;
    ndfOpen( NULL, name, mode, stat, &indf, &place, &status);
    if (raiseNDFException(&status)) return NULL;
    return NDF_create_object( indf, place );
};

// create a new NDF (simple) structure
static PyObject*
pyndf_new(NDF *self, PyObject *args)
{
	// use ultracam defaults
	const char *ftype = "_REAL";
	int ndim;
	PyObject* lb;
	PyObject* ub;
	if(!PyArg_ParseTuple(args, "siOO:pyndf_new", &ftype, &ndim, &lb, &ub))
		return NULL;
	if(ndim < 0 || ndim > 7)
		return NULL;
	// TODO: check for ftype here
	int status = SAI__OK;
	PyArrayObject* lower = (PyArrayObject*) PyArray_FROM_OTF(lb, NPY_UINT, NPY_IN_ARRAY | NPY_FORCECAST);
	PyArrayObject* upper = (PyArrayObject*) PyArray_FROM_OTF(ub, NPY_UINT, NPY_IN_ARRAY | NPY_FORCECAST);
	if (!lower || !upper)
		return NULL;
	if(PyArray_SIZE(lower) != ndim || PyArray_SIZE(upper) != ndim)
		return NULL;
        errBegin(&status);
        int indf = NDF__NOID;
	ndfNew(ftype,ndim,(int*)PyArray_DATA(lower),(int*)PyArray_DATA(upper),&self->_place,&indf,&status); // placeholder annulled by this routine
	Py_DECREF(lower);
	Py_DECREF(upper);
	if (raiseNDFException(&status))
		return NULL;
	return NDF_create_object( indf, NDF__NOPL);
}

// this copies a block of memory from a numpy array to a memory address
static PyObject*
pyndf_numpytoptr(NDF *self, PyObject *args)
{
	PyObject *npy, *ptrobj;
	PyArrayObject *npyarray;
	int el;
	size_t bytes;
	const char *ftype;
	if(!PyArg_ParseTuple(args, "OOis:pyndf_numpytoptr",&npy,&ptrobj,&el,&ftype))
		return NULL;
	void *ptr = NpyCapsule_AsVoidPtr(ptrobj);
	if (el <= 0 || ptr == NULL)
		return NULL;
	if(strcmp(ftype,"_INTEGER") == 0) {
		npyarray = (PyArrayObject*) PyArray_FROM_OTF(npy, NPY_INT, NPY_IN_ARRAY | NPY_FORCECAST);
		bytes = sizeof(int);
	} else if(strcmp(ftype,"_REAL") == 0) {
		npyarray = (PyArrayObject*) PyArray_FROM_OTF(npy, NPY_FLOAT, NPY_IN_ARRAY | NPY_FORCECAST);
		bytes = sizeof(float);
	} else if(strcmp(ftype,"_DOUBLE") == 0) {
		npyarray = (PyArrayObject*) PyArray_FROM_OTF(npy, NPY_DOUBLE, NPY_IN_ARRAY | NPY_FORCECAST);
		bytes = sizeof(double);
	} else if(strcmp(ftype,"_BYTE") == 0) {
		npyarray = (PyArrayObject*) PyArray_FROM_OTF(npy, NPY_BYTE, NPY_IN_ARRAY | NPY_FORCECAST);
		bytes = sizeof(char);
	} else if(strcmp(ftype,"_UBYTE") == 0) {
		npyarray = (PyArrayObject*) PyArray_FROM_OTF(npy, NPY_UBYTE, NPY_IN_ARRAY | NPY_FORCECAST);
		bytes = sizeof(char);
	} else {
		return NULL;
	}
	memcpy(ptr,PyArray_DATA(npyarray),el*bytes);
	Py_DECREF(npyarray);
	Py_RETURN_NONE;
}

// check an HDS type
inline int checkHDStype(const char *type)
{
	if(strcmp(type,"_INTEGER") != 0 && strcmp(type,"_REAL") != 0 && strcmp(type,"_DOUBLE") != 0 &&
			strcmp(type,"_LOGICAL") != 0 && strcmp(type,"_WORD") != 0 && strcmp(type,"UWORD") != 0 &&
			strcmp(type,"_BYTE") != 0 && strcmp(type,"_UBYTE") != 0 && strcmp(type,"_CHAR") != 0 &&
			strncmp(type,"_CHAR*",6) != 0)
		return 0;
	else
		return 1;
}

// create a new NDF extension
static PyObject*
pyndf_xnew(NDF *self, PyObject *args)
{
	int ndim = 0;
	const char *xname, *type;
	PyObject *dim;
	if(!PyArg_ParseTuple(args, "ss|iO:pyndf_xnew", &xname, &type, &ndim, &dim))
		return NULL;
	int status = SAI__OK;
	HDSLoc *loc = NULL;
	// perform checks if we're not making an extension header
	if(ndim != 0) {
		// check for HDS types
		if (!checkHDStype(type))
			return NULL;
		// need dims if it's not an ext
		if(ndim < 1 || dim == NULL)
			return NULL;
		PyArrayObject *npydim = (PyArrayObject*) PyArray_FROM_OTF(dim,NPY_INT,NPY_IN_ARRAY|NPY_FORCECAST);
		if (PyArray_SIZE(npydim) != ndim)
			return NULL;
                errBegin(&status);
		ndfXnew(self->_ndfid,xname,type,ndim,(int*)PyArray_DATA(npydim),&loc,&status);
		Py_DECREF(npydim);
	} else {
		// making an ext/struct
                errBegin(&status);
		ndfXnew(self->_ndfid,xname,type,0,0,&loc,&status);
	}
        if (raiseNDFException(&status)) return NULL;
	PyObject* pobj = NpyCapsule_FromVoidPtr(loc, PyDelLoc);
	return Py_BuildValue("O",pobj);
}

static PyObject*
pyndf_getbadpixval(NDF *self, PyObject *args)
{
	const char *type;
	if(!PyArg_ParseTuple(args, "s:pyndf_getpadpixval", &type))
		return NULL;
	if (strcmp(type,"_DOUBLE") == 0)
		return Py_BuildValue("f",VAL__BADD);
	else if (strcmp(type,"_REAL") == 0)
		return Py_BuildValue("f",VAL__BADR);
	else if (strcmp(type,"_INTEGER") == 0)
		return Py_BuildValue("i",VAL__BADI);
	else
		return NULL;
}

// map access to array component
static PyObject*
pyndf_map(NDF *self, PyObject* args)
{
	int el;
	void* ptr;
	const char* comp;
	const char* type;
	const char* mmod;
	if(!PyArg_ParseTuple(args,"sss:pyndf_map",&comp,&type,&mmod))
		return NULL;
	int status = SAI__OK;
	if(strcmp(comp,"DATA") != 0 && strcmp(comp,"QUALITY") != 0 &&
           strcmp(comp,"VARIANCE") != 0 && strcmp(comp,"ERROR") != 0) {
                PyErr_SetString( PyExc_ValueError, "Unsupported NDF data component" );
                return NULL;
        }
	if(strcmp(mmod,"READ") != 0 && strcmp(mmod,"UPDATE") != 0 &&
           strcmp(mmod,"WRITE") != 0) {
                PyErr_SetString( PyExc_ValueError, "Unsupported NDF update mode" );
		return NULL;
        }
	if(!checkHDStype(type)) {
                PyErr_SetString( PyExc_ValueError, "Unsupported HDS data type" );
		return NULL;
        }
	// can't use QUALITY with anything but a _UBYTE
	if(strcmp(comp,"QUALITY") == 0 && strcmp(type,"_UBYTE") != 0) {
                PyErr_SetString( PyExc_ValueError, "QUALITY requires type _UBYTE" );
		return NULL;
        }
        errBegin(&status);
	ndfMap(self->_ndfid,comp,type,mmod,&ptr,&el,&status);
	if (raiseNDFException(&status))
		return NULL;
	PyObject* ptrobj = NpyCapsule_FromVoidPtr(ptr,NULL);
	return Py_BuildValue("Oi",ptrobj,el);
}

// map access to array component
static PyObject*
pyndf_amap(NDF *self, PyObject* args)
{
	int el;
	void* ptr;
	const char* comp;
	const char* type;
	const char* mmod;
        int iaxis;
	if(!PyArg_ParseTuple(args,"siss:pyndf_amap",&comp,&iaxis,&type,&mmod))
		return NULL;
	int status = SAI__OK;
	if(strcmp(comp,"CENTRE") != 0 && strcmp(comp,"WIDTH") != 0 &&
           strcmp(comp,"VARIANCE") != 0 && strcmp(comp,"ERROR") != 0) {
                PyErr_SetString( PyExc_ValueError, "Unsupported NDF data component" );
                return NULL;
        }
	if(strcmp(mmod,"READ") != 0 && strcmp(mmod,"UPDATE") != 0 &&
           strcmp(mmod,"WRITE") != 0) {
                PyErr_SetString( PyExc_ValueError, "Unsupported NDF update mode" );
		return NULL;
        }
	if(!checkHDStype(type)) {
                PyErr_SetString( PyExc_ValueError, "Unsupported HDS data type" );
		return NULL;
        }
        errBegin(&status);
	ndfAmap(self->_ndfid,comp,iaxis,type,mmod,&ptr,&el,&status);
	if (raiseNDFException(&status))
		return NULL;
	PyObject* ptrobj = NpyCapsule_FromVoidPtr(ptr,NULL);
	return Py_BuildValue("Oi",ptrobj,el);
}

// unmap an NDF or mapped array
static PyObject*
pyndf_unmap(NDF* self, PyObject* args)
{
	const char* comp;
	if(!PyArg_ParseTuple(args,"s:pyndf_unmap",&comp))
		return NULL;
	int status = SAI__OK;
	if(strcmp(comp,"DATA") != 0 && strcmp(comp,"QUALITY") != 0 &&
			strcmp(comp,"VARIANCE") != 0 && strcmp(comp,"AXIS") != 0 &&
                        strcmp(comp,"*") != 0) {
                PyErr_SetString( PyExc_ValueError, "Unsupported NDF data component to unmap" );
		return NULL;
        }
        errBegin(&status);
	ndfUnmap(self->_ndfid,comp,&status);
	if (raiseNDFException(&status))
		return NULL;
	Py_RETURN_NONE;
}

// Reads an NDF into a numpy array
static PyObject*
pyndf_read(NDF *self, PyObject *args)
{
    int i;
    const char *comp;
    if(!PyArg_ParseTuple(args, "s:pyndf_read", &comp))
	return NULL;

    // series of declarations in an attempt to avoid problem with
    // goto fail
    const int MXLEN=32;
    char type[MXLEN+1];
    size_t nbyte;
    int npix, nelem;

    // Return None if component does not exist
    int state, status = SAI__OK;
    errBegin(&status);
    ndfState(self->_ndfid, comp, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    if(!state)
	Py_RETURN_NONE;

    PyArrayObject* arr = NULL;

    // Get dimensions, reverse order to account for C vs Fortran
    const int NDIMX = 10;
    int idim[NDIMX];
    npy_intp rdim[NDIMX];

    int ndim;
    ndfDim(self->_ndfid, NDIMX, idim, &ndim, &status);
    if (status != SAI__OK) goto fail;

    // Reverse order to account for C vs Fortran
    for(i=0; i<ndim; i++) rdim[i] = idim[ndim-i-1];

    // Determine the data type
    ndfType(self->_ndfid, comp, type, MXLEN+1, &status);
    if(status != SAI__OK) goto fail;

    // Create array of correct dimensions and type to save data to
    if(strcmp(type, "_REAL") == 0){
	arr = (PyArrayObject*) PyArray_SimpleNew(ndim, rdim, PyArray_FLOAT);
	nbyte = sizeof(float);
    }else if(strcmp(type, "_DOUBLE") == 0){
	arr = (PyArrayObject*) PyArray_SimpleNew(ndim, rdim, PyArray_DOUBLE);
	nbyte = sizeof(double);
    }else if(strcmp(type, "_INTEGER") == 0){
	arr = (PyArrayObject*) PyArray_SimpleNew(ndim, rdim, PyArray_INT);
	nbyte = sizeof(int);
    }else{
	PyErr_SetString(PyExc_IOError, "ndf_read error: unrecognised data type");
	goto fail;
    }
    if(arr == NULL) goto fail;

    // get number of elements, allocate space, map, store

    ndfSize(self->_ndfid, &npix, &status);
    if(status != SAI__OK) goto fail;
    void *pntr[1];
    ndfMap(self->_ndfid, comp, type, "READ", pntr, &nelem, &status);
    if(status != SAI__OK) goto fail;
    if(nelem != npix){
	PyErr_SetString(PyExc_IOError, "ndf_read error: number of elements different from number expected");
	goto fail;
    }
    memcpy(arr->data, pntr[0], npix*nbyte);
    ndfUnmap(self->_ndfid, comp, &status);
    if(status != SAI__OK) goto fail;

    return Py_BuildValue("N", PyArray_Return(arr));

fail:
    raiseNDFException(&status);
    Py_XDECREF(arr);
    return NULL;
};


static PyObject*
pyndf_state(NDF *self, PyObject *args)
{
    const char *comp;
    if(!PyArg_ParseTuple(args, "s:pyndf_state", &comp))
	return NULL;
    int state, status = SAI__OK;
    errBegin(&status);
    ndfState(self->_ndfid, comp, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("i", state);
};

static PyObject*
pyndf_xloc(NDF *self, PyObject *args)
{
    const char *xname, *mode;
    if(!PyArg_ParseTuple(args, "ss:pyndf_xloc", &xname, &mode))
	return NULL;
    HDSLoc* loc = NULL;
    int status = SAI__OK;
    errBegin(&status);
    ndfXloc(self->_ndfid, xname, mode, &loc, &status);
    if (raiseNDFException(&status)) return NULL;

    // PyCObject to pass pointer along to other wrappers
    PyObject *pobj = NpyCapsule_FromVoidPtr(loc, PyDelLoc);
    return Py_BuildValue("O", pobj);
};

static PyObject*
pyndf_xname(NDF *self, PyObject *args)
{
    int nex, nlen = 32;
    if(!PyArg_ParseTuple(args, "i|i:pyndf_xname", &nex, &nlen))
	return NULL;

    char xname[nlen+1];
    int status = SAI__OK;
    errBegin(&status);
    ndfXname(self->_ndfid, nex+1, xname, nlen+1, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("s", xname);
};

static PyObject*
pyndf_xnumb(NDF *self)
{
    int status = SAI__OK, nextn;
    errBegin(&status);
    ndfXnumb(self->_ndfid, &nextn, &status);
    if (raiseNDFException(&status)) return NULL;

    return Py_BuildValue("i", nextn);
};

static PyObject*
pyndf_xstat(NDF *self, PyObject *args)
{
    const char *xname;
    if(!PyArg_ParseTuple(args, "si:pyndf_xstat", &xname))
	return NULL;
    int state, status = SAI__OK;
    errBegin(&status);
    ndfXstat(self->_ndfid, xname, &state, &status);
    if (raiseNDFException(&status)) return NULL;

    return Py_BuildValue("i", state);
};

//
//
//  END OF METHODS - NOW DEFINE ATTRIBUTES AND MODULES
//

// Define the attributes

static PyMemberDef NDF_members[] = {
    {"_ndfid", T_INT, offsetof(NDF, _ndfid), 0,
     "NDF Identifier"},
    {"_place", T_INT, offsetof(NDF, _place), 0,
     "NDF Place holder"},
    {NULL}  /* Sentinel */
};


// The methods

static PyMethodDef NDF_methods[] = {

    {"acget", (PyCFunction)pyndf_acget, METH_VARARGS,
     "value = indf.acget(comp, iaxis) -- returns character component comp of axis iaxis (starts at 0), None if comp does not exist."},

    {"aform", (PyCFunction)pyndf_aform, METH_VARARGS,
     "value = indf.aform(comp, iaxis) -- returns storage form of an axis (iaxis starts at 0)."},

    {"annul", (PyCFunction)pyndf_annul, METH_NOARGS,
     "indf.annul() -- annuls the NDF identifier."},

    {"anorm", (PyCFunction)pyndf_anorm, METH_VARARGS,
     "state = indf.anorm(iaxis) -- determine axis normalisation flag (iaxis=-1 ORs all flags)."},

    {"aread", (PyCFunction)pyndf_aread, METH_VARARGS,
     "arr = indf.aread(comp,iaxis) -- reads component comp of axis iaxis. Returns None if does not exist"},

    {"astat", (PyCFunction)pyndf_astat, METH_VARARGS,
     "state = indf.astat(comp, iaxis) -- determine the state of an NDF axis component (iaxis starts at 0)."},

    {"init", (PyCFunction)pyndf_init, METH_NOARGS,
     "ndf.init() -- initialises the C ndf system."},

    {"begin", (PyCFunction)pyndf_begin, METH_NOARGS,
     "ndf.begin() -- starts a new NDF context."},

    {"bound", (PyCFunction)pyndf_bound, METH_NOARGS,
     "bound = indf.bound() -- returns pixel bounds, (2,ndim) array."},

    {"cget", (PyCFunction)pyndf_cget, METH_VARARGS,
     "value = indf.cget(comp) -- returns character component comp as a string, None if comp does not exist."},

    {"dim", (PyCFunction)pyndf_dim, METH_NOARGS,
     "dim = indf.dim() -- returns dimensions as 1D array."},

    {"end", (PyCFunction)pyndf_end, METH_NOARGS,
     "ndf.end() -- ends the current NDF context."},

    {"open", (PyCFunction)pyndf_open, METH_VARARGS,
     "indf = ndf.open(name) -- opens an NDF file."},

    {"read", (PyCFunction)pyndf_read, METH_VARARGS,
     "arr = indf.read(comp) -- reads component comp of an NDF (e.g. dat or var). Returns None if it does not exist."},

    {"state", (PyCFunction)pyndf_state, METH_VARARGS,
     "state = indf.state(comp) -- determine the state of an NDF component."},

    {"xloc", (PyCFunction)pyndf_xloc, METH_VARARGS,
     "loc = indf.xloc(xname, mode) -- return HDS locator."},

    {"xname", (PyCFunction)pyndf_xname, METH_VARARGS,
     "xname = indf.xname(n) -- return name of extension n (starting from 0)."},

    {"xnumb", (PyCFunction)pyndf_xnumb, METH_NOARGS,
     "nextn = indf.xnumb() -- return number of extensions."},

    {"xstat", (PyCFunction)pyndf_xstat, METH_VARARGS,
     "state = indf.xstat(xname) -- determine whether extension xname exists."},

    {"new", (PyCFunction)pyndf_new, METH_VARARGS,
     "ondf = indf.new(ftype,ndim,lbnd,ubnd) -- create a new simple ndf structure."},

    {"xnew", (PyCFunction)pyndf_xnew, METH_VARARGS,
     "loc = indf.xnew(xname,type,ndim,dim) -- create a new ndf extension."},

    {"map", (PyCFunction)pyndf_map, METH_VARARGS,
     "(pointer,elements) = indf.map(comp,type,mmod) -- map access to array component."},

    {"amap", (PyCFunction)pyndf_amap, METH_VARARGS,
     "(pointer,elements) = indf.amap(comp,iaxis,type,mmod) -- map access to axis array component."},

    {"unmap", (PyCFunction)pyndf_unmap, METH_VARARGS,
     "status = indf.unmap(comp) -- unmap an NDF or mapped NDF array."},

    {"ndf_numpytoptr", (PyCFunction)pyndf_numpytoptr, METH_VARARGS,
     "ndf_numpytoptr(array,pointer,elements,type) -- write numpy array to mapped pointer elements."},

    {"ndf_getbadpixval", (PyCFunction)pyndf_getbadpixval, METH_VARARGS,
     "ndf_getbadpixval(type) -- return a bad pixel value for given ndf data type."},

    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyTypeObject NDFType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "starlink.ndf",             /* tp_name */
    sizeof(NDF),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)NDF_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
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
    "Raw API for NDF access",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    NDF_methods,             /* tp_methods */
    NDF_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)NDF_init,      /* tp_init */
    0,                         /* tp_alloc */
    NDF_new,                 /* tp_new */
};

// Helper to create an object with an NDF identifier and placeholder

static PyObject *
NDF_create_object( int ndfid, int place )
{
  NDF * self = (NDF*)NDF_new( &NDFType, NULL, NULL );
  self->_ndfid = ndfid;
  self->_place = place;
  return (PyObject*)self;
}



#ifdef USE_PY3K

#define RETVAL m

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "ndf",
  "Raw NDF API",
  -1,
  NDF_methods,
  NULL,
  NULL,
  NULL,
  NULL
};

PyObject *PyInit_ndf(void)
#else

#define RETVAL

PyMODINIT_FUNC
initndf(void)
#endif
{
    PyObject *m = NULL;

    if (PyType_Ready(&NDFType) < 0)
        return RETVAL;

#ifdef USE_PY3K
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("ndf", NDF_methods,
                       "Raw NDF API");
#endif
    import_array();

    Py_INCREF(&NDFType);
    PyModule_AddObject(m, "ndf", (PyObject *)&NDFType);

    StarlinkNDFError = PyErr_NewException("starlink.ndf.error", NULL, NULL);
    Py_INCREF(StarlinkNDFError);
    PyModule_AddObject(m, "error", StarlinkNDFError);

    return RETVAL;
}
