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
#include "ndf/npy_3kcompat.h"

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
pydat_annul(PyObject *self, PyObject *args)
{
    PyObject* pobj;
    if(!PyArg_ParseTuple(args, "O:pydat_annul", &pobj))
	return NULL; 

    // Recover C-pointer passed via Python
    HDSLoc* loc = (HDSLoc*)NpyCapsule_AsVoidPtr(pobj);
    int status = SAI__OK;    
    errBegin(&status);
    datAnnul(&loc, &status);
    if(raiseNDFException(&status)) return NULL;
    Py_RETURN_NONE;
};

static PyObject* 
pydat_cell(PyObject *self, PyObject *args)
{
    PyObject *pobj1, *osub;
    if(!PyArg_ParseTuple(args, "OO:pydat_cell", &pobj1, &osub))
	return NULL; 

    // Recover C-pointer passed via Python
    HDSLoc* loc1 = (HDSLoc*)NpyCapsule_AsVoidPtr(pobj1);

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

    // PyCObject to pass pointer along to other wrappers
    PyObject *pobj2 = NpyCapsule_FromVoidPtr(loc2, PyDelLoc);
    Py_DECREF(sub);
    return Py_BuildValue("O", pobj2);

fail:    
    raiseNDFException(&status);
    Py_XDECREF(sub);
    return NULL;
};

static PyObject* 
pydat_index(PyObject *self, PyObject *args)
{
    PyObject* pobj;
    int index;
    if(!PyArg_ParseTuple(args, "Oi:pydat_index", &pobj, &index))
	return NULL; 

    // Recover C-pointer passed via Python
    HDSLoc* loc1 = (HDSLoc*)NpyCapsule_AsVoidPtr(pobj);
    HDSLoc* loc2 = NULL;

    int status = SAI__OK;    
    errBegin(&status);
    datIndex(loc1, index+1, &loc2, &status);
    if(raiseNDFException(&status)) return NULL;

    // PyCObject to pass pointer along to other wrappers
    PyObject *pobj2 = NpyCapsule_FromVoidPtr(loc2, PyDelLoc);
    return Py_BuildValue("O", pobj2);

};

static PyObject* 
pydat_find(PyObject *self, PyObject *args)
{
    PyObject* pobj1;
    const char* name;
    if(!PyArg_ParseTuple(args, "Os:pydat_find", &pobj1, &name))
	return NULL; 

    // Recover C-pointer passed via Python
    HDSLoc* loc1 = (HDSLoc*)NpyCapsule_AsVoidPtr(pobj1);
    HDSLoc* loc2 = NULL;

    int status = SAI__OK;    
    errBegin(&status);
    datFind(loc1, name, &loc2, &status);
    if (raiseNDFException(&status)) return NULL;

    // PyCObject to pass pointer along to other wrappers
    PyObject *pobj2 = NpyCapsule_FromVoidPtr(loc2, PyDelLoc);
    return Py_BuildValue("O", pobj2);
};

static PyObject* 
pydat_get(PyObject *self, PyObject *args)
{
    PyObject* pobj;
    if(!PyArg_ParseTuple(args, "O:pydat_get", &pobj))
	return NULL; 

    // Recover C-pointer passed via Python
    HDSLoc* loc = (HDSLoc*)NpyCapsule_AsVoidPtr(pobj);

    // guard against structures
    int state, status = SAI__OK;
    errBegin(&status);
    datStruc(loc, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    if(state){
	PyErr_SetString(PyExc_IOError, "dat_get error: cannot use on structures");
	return NULL;
    }

    // get type
    char typ_str[DAT__SZTYP+1];
    datType(loc, typ_str, &status);

    // get shape
    const int NDIMX=7;
    int ndim;
    hdsdim tdim[NDIMX];
    datShape(loc, NDIMX, tdim, &ndim, &status);
    if (raiseNDFException(&status)) return NULL;

    PyArrayObject* arr = NULL;

    // Either return values as a single scalar or a numpy array

    // Reverse order of dimensions
    npy_intp rdim[NDIMX];
    int i;
    for(i=0; i<ndim; i++) rdim[i] = tdim[ndim-i-1];

    if(strcmp(typ_str, "_INTEGER") == 0 || strcmp(typ_str, "_LOGICAL") == 0){
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
    return PyArray_Return(arr);

fail:    
    raiseNDFException(&status);
    Py_XDECREF(arr);
    return NULL;

};

static PyObject* 
pydat_name(PyObject *self, PyObject *args)
{
    PyObject* pobj;
    if(!PyArg_ParseTuple(args, "O:pydat_name", &pobj))
	return NULL; 

    // Recover C-pointer passed via Python
    HDSLoc* loc = (HDSLoc*)NpyCapsule_AsVoidPtr(pobj);

    char name_str[DAT__SZNAM+1];
    int status = SAI__OK;
    errBegin(&status);
    datName(loc, name_str, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("s", name_str);
};

static PyObject* 
pydat_ncomp(PyObject *self, PyObject *args)
{
    PyObject* pobj;
    if(!PyArg_ParseTuple(args, "O:pydat_ncomp", &pobj))
	return NULL; 

    // Recover C-pointer passed via Python
    HDSLoc* loc = (HDSLoc*)NpyCapsule_AsVoidPtr(pobj);

    int status = SAI__OK, ncomp;
    errBegin(&status);
    datNcomp(loc, &ncomp, &status);
    if (raiseNDFException(&status)) return NULL;

    return Py_BuildValue("i", ncomp);
};

static PyObject* 
pydat_shape(PyObject *self, PyObject *args)
{
    PyObject* pobj;
    if(!PyArg_ParseTuple(args, "O:pydat_type", &pobj))
	return NULL; 

    // Recover C-pointer passed via Python
    HDSLoc* loc = (HDSLoc*)NpyCapsule_AsVoidPtr(pobj);

    const int NDIMX=7;
    int ndim;
    hdsdim tdim[NDIMX];
    int status = SAI__OK;
    errBegin(&status);
    datShape(loc, NDIMX, tdim, &ndim, &status);
    if (raiseNDFException(&status)) return NULL;

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
pydat_state(PyObject *self, PyObject *args)
{
    PyObject* pobj;
    if(!PyArg_ParseTuple(args, "O:pydat_state", &pobj))
	return NULL; 

    // Recover C-pointer passed via Python
    HDSLoc* loc = (HDSLoc*)NpyCapsule_AsVoidPtr(pobj);

    int status = SAI__OK, state;
    errBegin(&status);
    datState(loc, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("i", state);
};

static PyObject* 
pydat_struc(PyObject *self, PyObject *args)
{
    PyObject* pobj;
    if(!PyArg_ParseTuple(args, "O:pydat_get", &pobj))
	return NULL; 

    // Recover C-pointer passed via Python
    HDSLoc* loc = (HDSLoc*)NpyCapsule_AsVoidPtr(pobj);

    // guard against structures
    int state, status = SAI__OK;
    errBegin(&status);
    datStruc(loc, &state, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("i", state);
};

static PyObject* 
pydat_type(PyObject *self, PyObject *args)
{
    PyObject* pobj;
    if(!PyArg_ParseTuple(args, "O:pydat_type", &pobj))
	return NULL; 

    // Recover C-pointer passed via Python
    HDSLoc* loc = (HDSLoc*)NpyCapsule_AsVoidPtr(pobj);

    char typ_str[DAT__SZTYP+1];
    int status = SAI__OK;
    errBegin(&status);
    datType(loc, typ_str, &status);
    if (raiseNDFException(&status)) return NULL;
    return Py_BuildValue("s", typ_str);
};

static PyObject* 
pydat_valid(PyObject *self, PyObject *args)
{
    PyObject* pobj;
    if(!PyArg_ParseTuple(args, "O:pydat_valid", &pobj))
	return NULL; 

    // Recover C-pointer passed via Python
    HDSLoc* loc = (HDSLoc*)NpyCapsule_AsVoidPtr(pobj);
    int state, status = SAI__OK;    
    errBegin(&status);
    datValid(loc, &state, &status);
    if (raiseNDFException(&status)) return NULL;

    return Py_BuildValue("i", state);
};

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


// make a new primitive
static PyObject*
pydat_new(PyObject *self, PyObject *args)
{
	PyObject *dimobj,*locobj;
	const char *type, *name;
	int ndim;
	if(!PyArg_ParseTuple(args, "OssiO:pydat_new", &locobj, &name, &type, &ndim, &dimobj))
		return NULL;
	HDSLoc* loc = (HDSLoc*)NpyCapsule_AsVoidPtr(locobj);
	if(!checkHDStype(type))
		return NULL;
	int status = SAI__OK;
        errBegin(&status);
	if (ndim > 0) {
		PyArrayObject *npydim = (PyArrayObject*) PyArray_FROM_OTF(dimobj,NPY_INT,NPY_IN_ARRAY|NPY_FORCECAST);
		hdsdim *dims = (hdsdim*)PyArray_DATA(npydim);
		datNew(loc,name,type,ndim,dims,&status);
		Py_DECREF(npydim);
	} else {
		datNew(loc,name,type,0,0,&status);
	}
	if (raiseNDFException(&status))
		return NULL;
	Py_RETURN_NONE;
}


// write a primitive
static PyObject*
pydat_put(PyObject *self, PyObject *args)
{
	PyObject *value, *locobj, *dimobj;
	PyArrayObject *npyval;
	const char* type;
	int ndim;
	if(!PyArg_ParseTuple(args,"OsiOO:pydat_put",&locobj,&type,&ndim,&dimobj,&value))
		return NULL;
	if(!checkHDStype(type))
		return NULL;
	HDSLoc* loc = (HDSLoc*)NpyCapsule_AsVoidPtr(locobj);
	// create a pointer to an array of the appropriate data type
	if(strcmp(type,"_INTEGER") == 0) {
		npyval = (PyArrayObject*) PyArray_FROM_OTF(value, NPY_INT, NPY_IN_ARRAY | NPY_FORCECAST);
	} else if(strcmp(type,"_REAL") == 0) {
		npyval = (PyArrayObject*) PyArray_FROM_OTF(value, NPY_FLOAT, NPY_IN_ARRAY | NPY_FORCECAST);
	} else if(strcmp(type,"_DOUBLE") == 0) {
		npyval = (PyArrayObject*) PyArray_FROM_OTF(value, NPY_DOUBLE, NPY_IN_ARRAY | NPY_FORCECAST);
	} else if(strcmp(type,"_BYTE") == 0) {
		npyval = (PyArrayObject*) PyArray_FROM_OTF(value, NPY_BYTE, NPY_IN_ARRAY | NPY_FORCECAST);
	} else if(strcmp(type,"_UBYTE") == 0) {
		npyval = (PyArrayObject*) PyArray_FROM_OTF(value, NPY_UBYTE, NPY_IN_ARRAY | NPY_FORCECAST);
	} else if(strncmp(type,"_CHAR*",6) == 0) {
		npyval = (PyArrayObject*) PyArray_FROM_OT(value, NPY_STRING);
	} else {
		return NULL;
	}
	void *valptr = PyArray_DATA(npyval);
	int status = SAI__OK;
        errBegin(&status);
	if (ndim > 0) {
		// npydim is 1-D array stating the size of each dimension ie. npydim = numpy.array([1072 1072])
		// these are stored in an hdsdim type (note these are declared as signed)
		PyArrayObject *npydim = (PyArrayObject*) PyArray_FROM_OTF(dimobj,NPY_INT,NPY_IN_ARRAY|NPY_FORCECAST);
		hdsdim *dims = (hdsdim*)PyArray_DATA(npydim);
		datPut(loc,type,ndim,dims,valptr,&status);
		Py_DECREF(npydim);
	} else {
		datPut(loc,type,0,0,valptr,&status);
	}
	if (raiseNDFException(&status))
		return NULL;
	Py_DECREF(npyval);
	Py_RETURN_NONE;
}

static PyObject*
pydat_putc(PyObject *self, PyObject *args)
{
	PyObject *strobj,*locobj;
	int strlen;
	if(!PyArg_ParseTuple(args,"OOi:pydat_putc",&locobj,&strobj,&strlen))
		return NULL;
	HDSLoc *loc = (HDSLoc*)NpyCapsule_AsVoidPtr(locobj);
	PyArrayObject *npystr = (PyArrayObject*) PyArray_FROM_OTF(strobj,NPY_STRING,NPY_FORCECAST);
	char *strptr = PyArray_DATA(npystr);
	int status = SAI__OK;
        errBegin(&status);
	datPutC(loc,0,0,strptr,(size_t)strlen,&status);
	if (raiseNDFException(&status))
		return NULL;
	Py_DECREF(npystr);
	Py_RETURN_NONE;
}


// The methods

static PyMethodDef HDSMethods[] = {

    {"dat_annul", pydat_annul, METH_VARARGS, 
     "dat_annul(loc) -- annuls the HDS locator."},

    {"dat_cell", pydat_cell, METH_VARARGS, 
     "loc2 = dat_cell(loc1, sub) -- returns locator of a cell of an array."},

    {"dat_index", pydat_index, METH_VARARGS, 
     "loc2 = dat_index(loc1, index) -- returns locator of index'th component (starts at 0)."},

    {"dat_find", pydat_find, METH_VARARGS, 
     "loc2 = dat_find(loc1, name) -- finds a named component, returns locator."},

    {"dat_get", pydat_get, METH_VARARGS, 
     "value = dat_get(loc) -- get data associated with locator regardless of type."},

    {"dat_name", pydat_name, METH_VARARGS, 
     "name_str = dat_name(loc) -- returns name of components."},

    {"dat_ncomp", pydat_ncomp, METH_VARARGS, 
     "ncomp = dat_ncomp(loc) -- return number of components."},

    {"dat_shape", pydat_shape, METH_VARARGS, 
     "dim = dat_shape(loc) -- returns shape of the component. dim=None for a scalar"},

    {"dat_state", pydat_state, METH_VARARGS, 
     "state = dat_state(loc) -- determine the state of an HDS component."},

    {"dat_struc", pydat_struc, METH_VARARGS, 
     "state = dat_struc(loc) -- is the component a structure."},

    {"dat_type", pydat_type, METH_VARARGS, 
     "typ_str = dat_type(loc) -- returns type of the component"},

    {"dat_valid", pydat_valid, METH_VARARGS, 
     "state = dat_valid(loc) -- is locator valid?"},


	{"dat_put", pydat_put, METH_VARARGS,
		"status = dat_put(loc,type,ndim,dim,value) -- write a primitive inside an ndf."},

	{"dat_new", pydat_new, METH_VARARGS,
		"dat_new(loc,name,type,ndim,dim) -- create a primitive given a locator."},

	{"dat_putc", pydat_putc, METH_VARARGS,
		"ndf_putc(loc,string) -- write a character string to primitive at locator."},

    {NULL, NULL, 0, NULL} /* Sentinel */
};

#ifdef USE_PY3K

#define RETVAL m

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "hds",
  NULL,
  -1,
  HDSMethods,
  NULL,
  NULL,
  NULL,
  NULL
};

PyObject *PyInit_hds(void)
#else

#define RETVAL

PyMODINIT_FUNC
init_hds(void)
#endif
{
    PyObject *m;

#ifdef USE_PY3K
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("hds", HDSMethods);
#endif
    import_array();

    StarlinkNDFError = PyErr_NewException("starlink.ndf.error", NULL, NULL);
    Py_INCREF(StarlinkNDFError);
    PyModule_AddObject(m, "error", StarlinkNDFError);

    return RETVAL;
}
