//
// Python/C interface file for ndf files

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

  if (*status == SAI__OK) return 0;

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
pyndf_acget(PyObject *self, PyObject *args)
{
    const char *comp;
    int indf, iaxis;
    if(!PyArg_ParseTuple(args, "isi:pyndf_acget", &indf, &comp, &iaxis))
	return NULL;

    // Return None if component does not exist
    int state, status = SAI__OK;
    int naxis = tr_iaxis(indf, iaxis, &status);
    errBegin(&status);
    ndfAstat(indf, comp, naxis, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    if(!state)
	Py_RETURN_NONE;

    int clen;
    ndfAclen(indf, comp, naxis, &clen, &status);
    char value[clen+1];
    ndfAcget(indf, comp, naxis, value, clen+1, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("s", value);
};

static PyObject* 
pyndf_aform(PyObject *self, PyObject *args)
{
    const char *comp;
    int indf, iaxis;
    if(!PyArg_ParseTuple(args, "isi:pyndf_aform", &indf, &comp, &iaxis))
	return NULL;
    int status = SAI__OK;
    int naxis = tr_iaxis(indf, iaxis, &status);
    char value[30];
    errBegin(&status);
    ndfAform(indf, comp, naxis, value, 30, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("s", value);
};

static PyObject* 
pyndf_annul(PyObject *self, PyObject *args)
{
    int indf;
    if(!PyArg_ParseTuple(args, "i:pyndf_annul", &indf))
	return NULL;
    int status = SAI__OK;
    errBegin(&status);
    ndfAnnul(&indf, &status);
    if (raiseNDFException(&status)) return NULL;
    Py_RETURN_NONE;
};

static PyObject* 
pyndf_anorm(PyObject *self, PyObject *args)
{
    int indf, iaxis;
    if(!PyArg_ParseTuple(args, "ii:pyndf_anorm", &indf, &iaxis))
	return NULL;
    int state, status = SAI__OK;
    int naxis = tr_iaxis(indf, iaxis, &status);
    errBegin(&status);
    ndfAnorm(indf, naxis, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("i", state);
};

static PyObject* 
pyndf_aread(PyObject *self, PyObject *args)
{
    int indf, iaxis;
    const char *MMOD = "READ";
    const char *comp;
    if(!PyArg_ParseTuple(args, "isi:pyndf_aread", &indf, &comp, &iaxis))
	return NULL;

    int status = SAI__OK;
    int naxis = tr_iaxis(indf, iaxis, &status);

    // Return None if component does not exist
    int state;
    errBegin(&status);
    ndfAstat(indf, comp, naxis, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    if(!state) Py_RETURN_NONE;

    // Get dimensions
    const int NDIMX = 10;
    int idim[NDIMX], ndim;
    ndfDim(indf, NDIMX, idim, &ndim, &status);
    if (raiseNDFException(&status)) return NULL;

    // get number for particular axis in question.
    int nelem = idim[naxis-1];

    // Determine the data type
    const int MXLEN=33;
    char type[MXLEN];
    ndfAtype(indf, comp, naxis, type, MXLEN, &status);
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
    ndfAmap(indf, comp, naxis, type, MMOD, pntr, &nread, &status);
    if (status != SAI__OK) goto fail;
    if(nelem != nread){
	printf("nread = %d, nelem = %d, iaxis = %d, %d\n",nread,nelem,iaxis,naxis);
	PyErr_SetString(PyExc_IOError, "ndf_aread error: number of elements different from number expected");
	goto fail;
    }
    memcpy(arr->data, pntr[0], nelem*nbyte);
    ndfAunmp(indf, comp, naxis, &status);

    return Py_BuildValue("N", PyArray_Return(arr));

fail:
    raiseNDFException(&status);
    Py_XDECREF(arr);
    return NULL;

};

static PyObject* 
pyndf_astat(PyObject *self, PyObject *args)
{
    const char *comp;
    int indf, iaxis;
    if(!PyArg_ParseTuple(args, "isi:pyndf_astat", &indf, &comp, &iaxis))
	return NULL;
    int state, status = SAI__OK;
    int naxis = tr_iaxis(indf, iaxis, &status);
    errBegin(&status);
    ndfAstat(indf, comp, naxis, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("i", state);
};

static PyObject* 
pyndf_init(PyObject *self, PyObject *args)
{
    int argc = 0, status = SAI__OK;
    char **argv = NULL;
    errBegin(&status);
    ndfInit(argc, argv, &status);
    if (raiseNDFException(&status)) return NULL;
    Py_RETURN_NONE;
};

static PyObject* 
pyndf_begin(PyObject *self, PyObject *args)
{
    ndfBegin();
    Py_RETURN_NONE;
};

static PyObject* 
pyndf_bound(PyObject *self, PyObject *args)
{
    int indf, i;
    if(!PyArg_ParseTuple(args, "i:pyndf_bound", &indf))
	return NULL;

    PyArrayObject* bound = NULL;
    int ndim;
    const int NDIMX=20;
    int *lbnd = (int *)malloc(NDIMX*sizeof(int));
    int *ubnd = (int *)malloc(NDIMX*sizeof(int));
    if(lbnd == NULL || ubnd == NULL)
	goto fail;

    int status = SAI__OK;
    errBegin(&status);
    ndfBound(indf, NDIMX, lbnd, ubnd, &ndim, &status ); 
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
pyndf_cget(PyObject *self, PyObject *args)
{
    const char *comp;
    int indf;
    if(!PyArg_ParseTuple(args, "is:pyndf_cget", &indf, &comp))
	return NULL;

    // Return None if component does not exist
    int state, status = SAI__OK;
    errBegin(&status);
    ndfState(indf, comp, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    if(!state)
	Py_RETURN_NONE;

    int clen;
    ndfClen(indf, comp, &clen, &status);
    if (raiseNDFException(&status)) return NULL;
    char value[clen+1];
    ndfCget(indf, comp, value, clen+1, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("s", value);
};

static PyObject* 
pyndf_dim(PyObject *self, PyObject *args)
{
    int indf, i;
    if(!PyArg_ParseTuple(args, "i:pyndf_dim", &indf))
	return NULL;

    PyArrayObject* dim = NULL;
    int ndim;
    const int NDIMX=20;
    int *idim = (int *)malloc(NDIMX*sizeof(int));
    if(idim == NULL)
	goto fail;

    int status = SAI__OK;
    errBegin(&status);
    ndfDim(indf, NDIMX, idim, &ndim, &status ); 
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
pyndf_end(PyObject *self, PyObject *args)
{
//    if(!PyArg_ParseTuple(args, "i:pyndf_end"))
//	return NULL;
    int status = SAI__OK;
    errBegin(&status);
    ndfEnd(&status);
    Py_RETURN_NONE;
};

// open an existing or new NDF file
static PyObject* 
pyndf_open(PyObject *self, PyObject *args)
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
    ndfOpen( NULL, name, mode, stat, &indf, &place, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("ii", indf, place);
};

// create a new NDF (simple) structure
static PyObject*
pyndf_new(PyObject *self, PyObject *args)
{
	// use ultracam defaults
	const char *ftype = "_REAL";
	int ndim, indf, place;
	PyObject* lb;
	PyObject* ub;
	if(!PyArg_ParseTuple(args, "iisiOO:pyndf_new", &indf, &place, &ftype, &ndim, &lb, &ub))
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
	ndfNew(ftype,ndim,(int*)PyArray_DATA(lower),(int*)PyArray_DATA(upper),&place,&indf,&status); // placeholder annulled by this routine
	if (raiseNDFException(&status))
		return NULL;
	Py_DECREF(lower);
	Py_DECREF(upper);
	return Py_BuildValue("i",indf);
}

// this copies a block of memory from a numpy array to a memory address
static PyObject*
pyndf_numpytoptr(PyObject *self, PyObject *args)
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
pyndf_xnew(PyObject *self, PyObject *args)
{
	int indf, ndim = 0;
	const char *xname, *type;
	PyObject *dim;
	if(!PyArg_ParseTuple(args, "iss|iO:pyndf_xnew", &indf, &xname, &type, &ndim, &dim))
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
		ndfXnew(indf,xname,type,ndim,(int*)PyArray_DATA(npydim),&loc,&status);
		Py_DECREF(npydim);
	} else {
		// making an ext/struct
                errBegin(&status);
		ndfXnew(indf,xname,type,0,0,&loc,&status);
	}
        if (raiseNDFException(&status)) return NULL;
	PyObject* pobj = NpyCapsule_FromVoidPtr(loc, PyDelLoc);
	return Py_BuildValue("O",pobj);
}

static PyObject*
pyndf_getbadpixval(PyObject *self, PyObject *args)
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
pyndf_map(PyObject *self, PyObject* args)
{
	int indf, el;
	void* ptr;
	const char* comp;
	const char* type;
	const char* mmod;
	if(!PyArg_ParseTuple(args,"isss:pyndf_map",&indf,&comp,&type,&mmod))
		return NULL;
	int status = SAI__OK;
	if(indf < 0)
		return NULL;
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
	ndfMap(indf,comp,type,mmod,&ptr,&el,&status);
	if (raiseNDFException(&status))
		return NULL;
	PyObject* ptrobj = NpyCapsule_FromVoidPtr(ptr,NULL);
	return Py_BuildValue("Oi",ptrobj,el);
}

// unmap an NDF or mapped array
static PyObject*
pyndf_unmap(PyObject* self, PyObject* args)
{
	int indf;
	const char* comp;
	if(!PyArg_ParseTuple(args,"is:pyndf_unmap",&indf,&comp))
		return NULL;
	int status = SAI__OK;
	if(indf < 0)
		return NULL;
	if(strcmp(comp,"DATA") != 0 && strcmp(comp,"QUALITY") != 0 &&
			strcmp(comp,"VARIANCE") != 0 && strcmp(comp,"AXIS") != 0 &&
                        strcmp(comp,"*") != 0) {
                PyErr_SetString( PyExc_ValueError, "Unsupported NDF data component to unmap" );
		return NULL;
        }
        errBegin(&status);
	ndfUnmap(indf,comp,&status);
	if (raiseNDFException(&status))
		return NULL;
	Py_RETURN_NONE;
}

// Reads an NDF into a numpy array
static PyObject* 
pyndf_read(PyObject *self, PyObject *args)
{
    int indf, i;
    const char *comp;
    if(!PyArg_ParseTuple(args, "is:pyndf_read", &indf, &comp))
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
    ndfState(indf, comp, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    if(!state)
	Py_RETURN_NONE;

    PyArrayObject* arr = NULL;

    // Get dimensions, reverse order to account for C vs Fortran
    const int NDIMX = 10;
    int idim[NDIMX];
    npy_intp rdim[NDIMX];

    int ndim;
    ndfDim(indf, NDIMX, idim, &ndim, &status);
    if (status != SAI__OK) goto fail; 

    // Reverse order to account for C vs Fortran
    for(i=0; i<ndim; i++) rdim[i] = idim[ndim-i-1];

    // Determine the data type
    ndfType(indf, comp, type, MXLEN+1, &status);
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

    ndfSize(indf, &npix, &status);
    if(status != SAI__OK) goto fail;
    void *pntr[1];
    ndfMap(indf, comp, type, "READ", pntr, &nelem, &status);
    if(status != SAI__OK) goto fail;
    if(nelem != npix){
	PyErr_SetString(PyExc_IOError, "ndf_read error: number of elements different from number expected");
	goto fail;
    }
    memcpy(arr->data, pntr[0], npix*nbyte);
    ndfUnmap(indf, comp, &status);
    if(status != SAI__OK) goto fail;

    return Py_BuildValue("N", PyArray_Return(arr));

fail:
    raiseNDFException(&status);
    Py_XDECREF(arr);
    return NULL;
};


static PyObject* 
pyndf_state(PyObject *self, PyObject *args)
{
    const char *comp;
    int indf;
    if(!PyArg_ParseTuple(args, "is:pyndf_state", &indf, &comp))
	return NULL;
    int state, status = SAI__OK;
    errBegin(&status);
    ndfState(indf, comp, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("i", state);
};

static PyObject* 
pyndf_xloc(PyObject *self, PyObject *args)
{
    const char *xname, *mode;
    int indf;
    if(!PyArg_ParseTuple(args, "iss:pyndf_xloc", &indf, &xname, &mode))
	return NULL;
    HDSLoc* loc = NULL;
    int status = SAI__OK;
    errBegin(&status);
    ndfXloc(indf, xname, mode, &loc, &status);
    if (raiseNDFException(&status)) return NULL;

    // PyCObject to pass pointer along to other wrappers
    PyObject *pobj = NpyCapsule_FromVoidPtr(loc, PyDelLoc);
    return Py_BuildValue("O", pobj);
};

static PyObject* 
pyndf_xname(PyObject *self, PyObject *args)
{
    int indf, nex, nlen = 32;
    if(!PyArg_ParseTuple(args, "ii|i:pyndf_xname", &indf, &nex, &nlen))
	return NULL;

    char xname[nlen+1];
    int status = SAI__OK;
    errBegin(&status);
    ndfXname(indf, nex+1, xname, nlen+1, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("s", xname);
};

static PyObject* 
pyndf_xnumb(PyObject *self, PyObject *args)
{
    int indf;
    if(!PyArg_ParseTuple(args, "i:pyndf_xnumb", &indf))
	return NULL;
    int status = SAI__OK, nextn;
    errBegin(&status);
    ndfXnumb(indf, &nextn, &status);
    if (raiseNDFException(&status)) return NULL;

    return Py_BuildValue("i", nextn);
};

static PyObject* 
pyndf_xstat(PyObject *self, PyObject *args)
{
    const char *xname;
    int indf;
    if(!PyArg_ParseTuple(args, "isi:pyndf_xstat", &indf, &xname))
	return NULL;
    int state, status = SAI__OK;
    errBegin(&status);
    ndfXstat(indf, xname, &state, &status);
    if (raiseNDFException(&status)) return NULL;

    return Py_BuildValue("i", state);
};

// The methods

static PyMethodDef NdfMethods[] = {


    {"ndf_acget", pyndf_acget, METH_VARARGS, 
     "value = ndf_acget(indf, comp, iaxis) -- returns character component comp of axis iaxis (starts at 0), None if comp does not exist."},

    {"ndf_aform", pyndf_aform, METH_VARARGS, 
     "value = ndf_aform(indf, comp, iaxis) -- returns storage form of an axis (iaxis starts at 0)."},

    {"ndf_annul", pyndf_annul, METH_VARARGS, 
     "ndf_annul(indf) -- annuls the NDF identifier."},

    {"ndf_anorm", pyndf_anorm, METH_VARARGS, 
     "state = ndf_anorm(indf, iaxis) -- determine axis normalisation flag (iaxis=-1 ORs all flags)."},

    {"ndf_aread", pyndf_aread, METH_VARARGS, 
     "arr = ndf_aread(indf,comp,iaxis) -- reads component comp of axis iaxis. Returns None if does not exist"},

    {"ndf_astat", pyndf_astat, METH_VARARGS, 
     "state = ndf_astat(indf, comp, iaxis) -- determine the state of an NDF axis component (iaxis starts at 0)."},

    {"ndf_init", pyndf_init, METH_VARARGS, 
     "ndf_init() -- initialises the C ndf system."},

    {"ndf_begin", pyndf_begin, METH_VARARGS, 
     "ndf_begin() -- starts a new NDF context."},

    {"ndf_bound", pyndf_bound, METH_VARARGS, 
     "bound = ndf_bound(indf) -- returns pixel bounds, (2,ndim) array."},

    {"ndf_cget", pyndf_cget, METH_VARARGS, 
     "value = ndf_cget(indf, comp) -- returns character component comp as a string, None if comp does not exist."},

    {"ndf_dim", pyndf_dim, METH_VARARGS, 
     "dim = ndf_dim(indf) -- returns dimensions as 1D array."},

    {"ndf_end", pyndf_end, METH_VARARGS, 
     "ndf_end() -- ends the current NDF context."},

    {"ndf_open", pyndf_open, METH_VARARGS, 
     "(indf,place) = ndf_open(name) -- opens an NDF file."},

    {"ndf_read", pyndf_read, METH_VARARGS, 
     "arr = ndf_read(indf,comp) -- reads component comp of an NDF (e.g. dat or var). Returns None if it does not exist."},

    {"ndf_state", pyndf_state, METH_VARARGS, 
     "state = ndf_state(indf, comp) -- determine the state of an NDF component."},

    {"ndf_xloc", pyndf_xloc, METH_VARARGS, 
     "loc = ndf_xloc(indf, xname, mode) -- return HDS locator."},

    {"ndf_xname", pyndf_xname, METH_VARARGS, 
     "xname = ndf_xname(indf, n) -- return name of extension n (starting from 0)."},

    {"ndf_xnumb", pyndf_xnumb, METH_VARARGS, 
     "nextn = ndf_xnumb(indf) -- return number of extensions."},

    {"ndf_xstat", pyndf_xstat, METH_VARARGS, 
     "state = ndf_xstat(indf, xname) -- determine whether extension xname exists."},

	{"ndf_new", pyndf_new, METH_VARARGS,
		"(place,indf) = ndf_new(ftype,ndim,lbnd,ubnd,place) -- create a new simple ndf structure."},

	{"ndf_xnew", pyndf_xnew, METH_VARARGS,
		"loc = ndf_xnew(indf,xname,type,ndim,dim) -- create a new ndf extension."},

	{"ndf_map", pyndf_map, METH_VARARGS,
		"(pointer,elements) = ndf_map(indf,comp,type,mmod) -- map access to array component."},

	{"ndf_unmap", pyndf_unmap, METH_VARARGS,
		"status = ndf_unmap(indf,comp) -- unmap an NDF or mapped NDF array."},

	{"ndf_numpytoptr", pyndf_numpytoptr, METH_VARARGS,
		"ndf_numpytoptr(array,pointer,elements,type) -- write numpy array to mapped pointer elements."},

	{"ndf_getbadpixval", pyndf_getbadpixval, METH_VARARGS,
		"ndf_getbadpixval(type) -- return a bad pixel value for given ndf data type."},

    {NULL, NULL, 0, NULL} /* Sentinel */
};

#ifdef USE_PY3K

#define RETVAL m

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_ndf",
  NULL,
  -1,
  NdfMethods,
  NULL,
  NULL,
  NULL,
  NULL
};

PyObject *PyInit__ndf(void)
#else

#define RETVAL

PyMODINIT_FUNC
init_ndf(void)
#endif
{
    PyObject *m;

#ifdef USE_PY3K
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("_ndf", NdfMethods);
#endif
    import_array();

    StarlinkNDFError = PyErr_NewException("starlink.ndf.error", NULL, NULL);
    Py_INCREF(StarlinkNDFError);
    PyModule_AddObject(m, "error", StarlinkNDFError);

    return RETVAL;
}
