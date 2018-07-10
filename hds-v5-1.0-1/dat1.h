
#ifndef DAT1_H_INCLUDED
#define DAT1_H_INCLUDED

/*
*+
*  Name:
*     dat1.h

*  Purpose:
*     Internal support routines for data access

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Include file

*  Invocation:
*     #include "dat1.h"

*  Description:
*     Defines internal interfaces for data access layer.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-08-15 (TIMJ):
*        Initial version
*     {enter_further_changes_here}

*  Copyright:
*     Copyright (C) 2014 Cornell University
*     All Rights Reserved.

*  Licence:
*     Redistribution and use in source and binary forms, with or
*     without modification, are permitted provided that the following
*     conditions are met:
*
*     - Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*
*     - Redistributions in binary form must reproduce the above
*       copyright notice, this list of conditions and the following
*       disclaimer in the documentation and/or other materials
*       provided with the distribution.
*
*     - Neither the name of the {organization} nor the names of its
*       contributors may be used to endorse or promote products
*       derived from this software without specific prior written
*       permission.
*
*     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
*     CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
*     INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
*     MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
*     CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
*     SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
*     LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
*     USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
*     AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
*     IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
*     THE POSSIBILITY OF SUCH DAMAGE.

*  Bugs:
*     {note_any_bugs_here}
*-
*/

#include <pthread.h>
#include "hdf5.h"
#include "hdf5_hl.h"
#include "hds1.h"
#include "hds_types.h"

/* Access mode requested for a particular primitive type */
typedef enum {
  HDSMODE_UNKNOWN = 0,
  HDSMODE_READ,
  HDSMODE_WRITE,
  HDSMODE_UPDATE
} hdsmode_t;

/* All the standard HDS types as an enum. For internal use only. */
typedef enum {
  HDSTYPE_NONE = 0,
  HDSTYPE_BYTE,
  HDSTYPE_UBYTE,
  HDSTYPE_WORD,
  HDSTYPE_UWORD,
  HDSTYPE_INTEGER,
  HDSTYPE_INT64,
  HDSTYPE_REAL,
  HDSTYPE_DOUBLE,
  HDSTYPE_LOGICAL,
  HDSTYPE_CHAR,
  HDSTYPE_STRUCTURE
} hdstype_t;

/* Which shell should be used when expanding environment
   variables. Not all HDS supported shells are supported
   by this library.
*/

typedef enum {
  HDS__NOSHELL = -1,  /* Do not expand shell variables */
  HDS__SHSHELL,       /* Use "sh" aka wordexp() */
  HDS__MAXSHELL       /* Too high */
} hds_shell_t;

/* Global Constants:                                                        */
/* ================                                                         */
#include "dat_par.h"

/* Internal Constants */
#define DAT__MXCHR 0xffff        /* Max characters in a character data type */

/* Internal definitions of true and false */
#define HDS_TRUE 1
#define HDS_FALSE 0

#define HDS_ISTRUE(X) ( X )
#define HDS_ISFALSE(X) ( !( X ) )


 /* Prefix to use for arrays of structures. We deliberately make it longer
    DAT__SZNAM and more verbose to make it impossible to be present by
    chance from the user side. */
#define DAT__CELLNAME "ARRAY_OF_STRUCTURES_CELL"

/* Names of attributes */
#define HDS__ATTR_STRUCT_TYPE "CLASS"
#define HDS__ATTR_STRUCT_DIMS "HDS_STRUCTURE_DIMS"
#define HDS__ATTR_ROOT_NAME "HDS_ROOT_NAME"
#define HDS__ATTR_ROOT_PRIMITIVE "HDS_ROOT_IS_PRIMITIVE"

/* This structure  contains information about an HDF5 object (group or
   dataset) that is common to all the locators that refer to the object. */
typedef struct Handle {
   pthread_mutex_t mutex;   /* Guards access to the values in the handle */

   int nwrite_lock;         /* Number of current write locks (0 or 1) */
   pthread_t write_locker;  /* ID for thread holding write lock */
   int nread_lock;          /* Number of current read locks (0 or more) */
   pthread_t *read_lockers; /* Array of IDs for thread holding read locks */
   int maxreaders;          /* Current size of "read_lockers" array */

   struct Handle *parent;   /* Pointer to Handle describing the parent object */
   struct Handle **children;/* Pointer to array holding pointers to Handles
                                for any known child objects */
   int nchild;              /* The length of the "children" array */
   char *name;              /* Name (cleaned) of the HDF object within its parent */
   char docheck;            /* If non-zero, check any lock is appropriate
                               before using the locator */
} Handle;

/* Private definition of the HDS locator struct */
typedef struct LOC {
  int hds_version;   /* Implementation version number. Always 5 at the moment.
                        Note, this MUST be the first field in the structure. */
  Handle *handle;    /* Structure holding fixed info for the HDF object */
  void *pntr;        /* Pointer to memory mapped data array [datMap only] */
  void *regpntr;     /* Pointer that was registered with CNF */
  size_t bytesmapped;/* Number of bytes mapped into memory [datMap only] */
  size_t vectorized; /* Non-zero if vectorized */
  hid_t file_id;     /* Set if this locator is associated with a root file */
  hid_t dataset_id;  /* Set if this is a dataset "primitive type" */
  hid_t dataspace_id;/* Set if this is a primitive with dimensions */
  hid_t group_id;    /* Set if this locator is associated with a group */
  hid_t dtype;       /* Set if a special data type was created for this locator */
  hdsmode_t accmode; /* Access mode for memory mapped data [datMap only] */
  int ndims;         /* Number of dimensions in mapdims [datMap only] */
  hdsdim mapdims[DAT__MXDIM]; /* Dimensionality of mapped dims [datMap only] */
  hdsbool_t iscell;  /* Is this a single cell? */
  hdsbool_t isslice; /* Is this a slice? */
  hdsbool_t isprimary;/* Is this a primary locator (and so owns its own file_id) */
  hdsbool_t isdiscont;/* Is this a discontiguous slice? */
  hdsbool_t uses_true_mmap;  /* Indicates that we have true mmap [datMap only] */
  int fdmap;  /* File descriptor for mapped data (can free if >0) [datMap only] */
  char maptype[DAT__SZTYP+1]; /* HDS type string used for memory mapping [datMap only] */
  char grpname[DAT__SZGRP+1]; /* Name of group associated with locator */
} HDSLoc;


/* This structure contains information about data types.
   Values are obtained by use dat1TypeInfo(). */
typedef struct HdsTypeInfo {
  double BADD;
  int64_t BADK;
  float BADR;
  int BADI;
  hdsbool_t BADL;
  short BADW;
  unsigned short BADUW;
  char BADB;
  unsigned char BADUB;
  char BADC;
} HdsTypeInfo;


/* Wrapper to check HDF5 status and map to EMS.
   Also does not call the routine unless status is good.
   Assumes inherited status is available in "status" and
   that there is a CLEANUP label. Use CALLHDF if the function
   return type can be handled by a herr_t type.
 */

#define CALLHDFE( dtype, retval, hfunc, errcode, errfunc ) \
  retval = 0;                                           \
  if (*status == SAI__OK) {                             \
    dtype h5err;                                        \
    h5err = hfunc;                                      \
    if (h5err >= 0) {                                   \
      retval = h5err;                                   \
    } else {                                            \
      *status = errcode;                                \
      dat1H5EtoEMS( status );                           \
      errfunc;                                          \
      goto CLEANUP;                                     \
    }                                                   \
  }

  /* Simpler form of CALLHDFE that does not require the
     return type to be specified */
#define CALLHDF( retval, hfunc, errcode, errfunc )  CALLHDFE( herr_t, retval, hfunc, errcode, errfunc )

/* Simpler quick wrapper when we do not care about the return value
   or including an explicit error message */
#define CALLHDFQ( hfunc )                                               \
  if (*status == SAI__OK) {                                             \
    herr_t h5err;                                                       \
    h5err = hfunc;                                                      \
    if (h5err < 0) {                                                    \
      *status = DAT__HDF5E;                                             \
      dat1H5EtoEMS( status );                                           \
      emsRepf("HDF5ERR","Error calling HDF5 with '%s'", status, "" #hfunc ""); \
      goto CLEANUP;                                                     \
    }                                                                   \
  }

/* Internal API prototypes */

void dau1CheckName( const char * name, int isname, char * buf,
                    size_t buflen, int * status );

char * dau1CheckFileName( const char * file_str, int *status );

int dau1CheckType ( hdsbool_t asmem, const char * type_str, hid_t * h5type,
                    char * norm_str, size_t normlen, int * status );

void dat1InitHDF5(void);

void dat1H5EtoEMS( int *status );

HDSLoc *
dat1New( const HDSLoc *locator, hdsbool_t isprimary, const char *name_str, const char *type_str,
        int ndim, const hdsdim dims[], int *status);

void
dat1NewPrim( hid_t group_id, int ndim, const hsize_t h5dims[], hid_t h5type,
             const char * name_str, hid_t * dataset_id, hid_t *dataspace_id, int *status );

hid_t dat1RetrieveContainer( const HDSLoc *locator, int * status );
hid_t dat1RetrieveIdentifier( const HDSLoc * locator, int * status );

HDSLoc * dat1FreeLoc( HDSLoc * locator, int * status );
HDSLoc * dat1AllocLoc( int * status );
void dat1DumpLoc( const HDSLoc * locator, int * status );

hdstype_t
dat1Type( const HDSLoc *locator, int * status );

int
dat1IsStructure( const HDSLoc * locator, int * status );

void
dat1ExportDims( int ndims, const hsize_t h5dims[], hdsdim hdsdims[],
                int *status );

void
dat1ImportDims( int ndims, const hdsdim hdsdims[], hsize_t h5dims[],
                int *status );

char *
dat1GetFullName( hid_t objid, int asfile, ssize_t * namlen, int *status);

void
dat1Coords2CellName( int ndim, const hdsdim coords[], char * cellname,
                     size_t cellnamelen, int * status );

char * dat1FixNameCell( const char * instr, int * status );

HDSLoc *
dat1ImportFloc ( const char flocator[DAT__SZLOC], int loc_length, int * status);

void
dat1EncodeSubscript( int ndim, hdsbool_t canbecell, const hdsdim lower[], const hdsdim upper[],
                     char *buf, size_t buflen, int *status );

void
dat1Index2Coords ( size_t idx, int ndim, const hdsdim arraydims[DAT__MXDIM],
                   hdsdim coords[DAT__MXDIM], int *status );

hdstype_t
dau1HdsType( hid_t h5type, int * status );

HdsTypeInfo *
dat1TypeInfo( void );

int
dat1CvtChar( size_t nval, hdstype_t intype, size_t nbin,
             hdstype_t outtype, size_t nbout, const void * imp, void * exp,
             size_t *nbad, int * status );

int
dat1CvtLogical( size_t nval, hdstype_t intype, size_t nbin,
             hdstype_t outtype, size_t nbout, const void * imp, void * exp,
             size_t *nbad, int * status );

int
dat1GetBounds( const HDSLoc * locator, hdsdim lower[DAT__MXDIM],
               hdsdim upper[DAT__MXDIM], hdsbool_t * issubset,
               int *actdim, int * status );

int
dat1GetDataDims( const HDSLoc * locator, hdsdim dims[DAT__MXDIM],
                 int *actdim, int * status );

hid_t
dau1Native2MemType( hid_t nativetype, int * status );

hdsbool_t
hds1RemoveLocator( const HDSLoc * loc, int *status );

void
dat1SetStructureDims( hid_t group_id, int ndim, const hdsdim dims[], int *status );

hid_t
dat1CreateStructureCell( hid_t group_id, size_t index, const char * typestr, const char * parentstr,
                         int ndim, const hdsdim dims[], int *status );

int
hds1RegLocator(HDSLoc *locator, int *status);

int
hds1FlushFile( hid_t file_id, int *status);

hdsbool_t
hds1UnregLocator( HDSLoc * loc, int *status );

size_t
hds1PrimaryCount( hid_t file_id, int * status );

void
hds1ShowFiles( hdsbool_t listfiles, hdsbool_t listlocs, int * status );

void
hds1ShowLocators( hid_t file_id, int * status );

int
hds1CountFiles();

int
hds1CountLocators( size_t ncomp, char **comps, hdsbool_t skip_scratch_root, int * status );

Handle *
hds1FindHandle( hid_t file_id, int *status );

void
dat1SetAttrString( hid_t obj_id, const char * attrname,
                   const char * value, int * status );

void
dat1SetAttr( hid_t obj_id, const char * attrname, hid_t attrtype,
             size_t nvals, const void * value, int * status );

void
dat1SetAttrInt( hid_t obj_id, const char * attrname,
                int value, int * status );

void
dat1SetAttrBool( hid_t obj_id, const char * attrname,
                 hdsbool_t value, int * status );

void
dat1SetAttrHdsdims( hid_t obj_id, const char * attrname,
                    int nvals, const hdsdim values[], int * status );

hdsbool_t
dat1GetAttr( hid_t obj_id, const char * attrname, hid_t attrtype,
             size_t maxvals, void * values, size_t *actvals, int * status );

hdsbool_t
dat1GetAttrBool( hid_t objid, const char * attrname, hdsbool_t usedef,
                 hdsbool_t defval, int *status);

void
dat1GetAttrHdsdims( hid_t objid, const char * attrname, hdsbool_t usedef,
                    size_t ndefault, const hdsdim defdims[], size_t maxvals,
                    hdsdim values[], size_t *actvals, int *status);

int
dat1GetAttrInt( hid_t objid, const char * attrname, hdsbool_t usedef,
                 int defval, int *status);

void
dat1GetAttrString( hid_t objid, const char * attrname, hdsbool_t usedef,
                   const char * defval, char * attrval, size_t attrvallen,
                   int *status);

void dat1Getenv( const char *varname, int def, int *val );

hdsbool_t hds1GetUseMmap();

hds_shell_t hds1GetShell();

int dat1Annul( HDSLoc *locator, int * status );

hid_t dat1GetParentID( hid_t objid, hdsbool_t allow_root, int *status );

int
dat1GetStructureDims( const HDSLoc * locator, int maxdims, hdsdim dims[], int *status );

hdsbool_t
dat1NeedsRootName( hid_t objid, hdsbool_t wantprim, char * rootname, size_t rootnamelen, int * status );

Handle *dat1Handle( const HDSLoc *parent_loc, const char *name, int
rdonly, int * status );
Handle *dat1EraseHandle( Handle *parent, const char *name, int * status );
Handle *dat1FreeHandle( Handle *handle );
int dat1ValidateLocator( const char *func, int checklock, const HDSLoc *loc, int rdonly, int *status );
Handle *dat1HandleLock( Handle *handle, int oper, int recurs, int rdonly, int *result, int *status );
void dat1HandleMsg( const char *token, const Handle *handle );

/* DAT1_H_INCLUDED */
#endif
