/*
*+
*  Name:
*     datGet

*  Purpose:
*     Read primitive

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     int datGet(const HDSLoc *locator, const char *type_str, int ndim,
*                const hdsdim dims[], void *values, int *status);

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Locator from which to obtain data
*     type_str = const char * (Given)
*        Data type to use for the read. Type conversion will be
*        performed if the underyling data type is different.
*     ndim = int (Given)
*        Number of dimensions in receiving data buffer.
*     dims = const hdsdim [] (Given)
*        Dimensionality of receiving data buffer.
*     values = void * (Given and Returned)
*        Buffer to receive data files.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Read data from a locator, performing type conversion as
*     required, and store it in the supplied buffer.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     DSB: David S Berry (EAO)
*     {enter_new_authors_here}

*  Notes:
*

*  History:
*     2014-08-28 (TIMJ):
*        Initial version
*     2017-05-24 (DSB):
*        Report an error if the supplied dimensions are different to the
*        shape of the supplied object.
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

#include "hdf5.h"

#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"
#include "dat_err.h"

int
datGet(const HDSLoc *locator, const char *type_str, int ndim,
       const hdsdim dims[], void *values, int *status) {

  char datatypestr[DAT__SZTYP+1];
  char namestr[DAT__SZNAM+1];
  char normtypestr[DAT__SZTYP+1];
  hdsdim locdims[DAT__MXDIM];
  hdstype_t doconv = HDSTYPE_NONE;
  hdstype_t intype = HDSTYPE_NONE;
  hdstype_t outtype = HDSTYPE_NONE;
  hid_t h5type = 0;
  hid_t mem_dataspace_id = 0;
  hsize_t h5dims[DAT__MXDIM];
  int actdim;
  int defined = 0;
  int i;
  int isprim;
  size_t nbin = 0;
  size_t nbout = 0;
  size_t nelem = 0;
  void * tmpvalues = NULL;

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datGet", 1, locator, 1, status );

  /* For error messages */
  datName( locator, namestr, status);
  datType( locator, datatypestr, status );

  /* Convert the HDS data type to HDF5 data type */
  isprim = dau1CheckType( 1, type_str, &h5type, normtypestr,
                          sizeof(normtypestr), status );

  if (!isprim) {
    if (*status == SAI__OK) {
      *status = DAT__TYPIN;
      emsRepf("datGet_1", "datGet: Data type must be a primitive type and not '%s'",
              status, normtypestr);
    }
    goto CLEANUP;
  }

  /* Get the shape of the supplied object. */
  datShape( locator, DAT__MXDIM, locdims, &actdim, status );

  /* Check the supplied dimensions are correct. */
  if( *status == SAI__OK ) {
    if( ndim == actdim ) {
      for( i = 0; i < ndim; i++ ) {
        if( locdims[i] != dims[i] ) {
          *status = DAT__DIMIN;
          emsRepf("", "datGet: Supplied dimension (%" HDS_DIM_FORMAT
                  ") on axis %d is incorrect - it should be %"
                  HDS_DIM_FORMAT ".", status, dims[i], i+1, locdims[i] );
          break;
        }
      }
    } else {
      *status = DAT__DIMIN;
      emsRepf("", "datGet: Supplied no. of axes (%d) is incorrect - it "
              "should be %d.", status, ndim, actdim );
    }
  }

  if (*status != SAI__OK) goto CLEANUP;


  /* Ensure that the locator is defined */
  datState( locator, &defined, status );
  if (!defined) {
    *status = DAT__UNSET;
    emsRep("datGet_1b", "datGet: Primitive object is undefined. Nothing to get.",
           status );
    goto CLEANUP;
  }

 /* Check data types and do conversion if required */
  intype = dat1Type( locator, status );
  outtype = dau1HdsType( h5type, status );

  if ((outtype == HDSTYPE_CHAR && intype != HDSTYPE_CHAR) ||
      (outtype != HDSTYPE_CHAR && intype == HDSTYPE_CHAR)) {
    doconv = HDSTYPE_CHAR;
  } else if ((outtype == HDSTYPE_LOGICAL && intype != HDSTYPE_LOGICAL) ||
             (outtype != HDSTYPE_LOGICAL && intype == HDSTYPE_LOGICAL)) {
    doconv = HDSTYPE_LOGICAL;
  }

  if ( doconv == HDSTYPE_LOGICAL || doconv == HDSTYPE_CHAR ) {
    /* We need to do the conversion because HDF5 does not seem
       to be able to convert numerical to string or string
       to numerical types internally. HDS has always been able
       to do so. Also, the number <=> bitfield mapping does not
       seem to be compatible with HDS so we do our own _LOGICAL handling. */
    /* First we allocate temporary space, then read the data
       from HDF5 in native form */
    hid_t tmptype = 0;

    /* Number of elements to convert */
    datSize( locator, &nelem, status );

    /* Number of bytes per element in the input (on disk) type */
    datLen( locator, &nbin, status );

    /* Number of bytes per element in the output (in memory) type */
    CALLHDF(nbout,
            H5Tget_size( h5type ),
            DAT__HDF5E,
            emsRep("datPut_size", "datPut: Error obtaining size of input type",
                   status)
            );

    /* Create a buffer to receive the converted values */
    tmpvalues = MEM_MALLOC( nelem * nbin );

    /* The type of the things we are reading has now changed
       so we need to update that */
    if (h5type) H5Tclose(h5type);
    CALLHDF( h5type,
             H5Dget_type( locator->dataset_id ),
             DAT__HDF5E,
             emsRep("datPut_type", "datGet: Error obtaining data type of native dataset", status)
             );

    tmptype = dau1Native2MemType( h5type, status );
    H5Tclose(h5type);
    h5type = tmptype;
  }

  /* Copy dimensions if appropriate */
  dat1ImportDims( ndim, dims, h5dims, status );

  /* Create a memory dataspace for the incoming data */
  CALLHDF( mem_dataspace_id,
           H5Screate_simple( ndim, h5dims, NULL),
           DAT__HDF5E,
           emsRepf("datGet_2", "datGet: Error allocating in-memory dataspace for object %s",
                   status, namestr )
           );

  CALLHDFQ( H5Dread( locator->dataset_id, h5type, mem_dataspace_id,
                     locator->dataspace_id, H5P_DEFAULT,
                     (tmpvalues ? tmpvalues : values ) ) );

  if (tmpvalues) {
    /* Now convert from what we have read to what we need */
    size_t nbad = 0;
    if (doconv == HDSTYPE_CHAR) {
      dat1CvtChar( nelem, intype, nbin, outtype, nbout, tmpvalues,
                   values, &nbad, status );
    } else if (doconv == HDSTYPE_LOGICAL) {
      dat1CvtLogical( nelem, intype, nbin, outtype, nbout, tmpvalues,
                      values, &nbad, status );
    } else {
      if (*status != SAI__OK) {
        *status = DAT__TYPIN;
        emsRep("datGet_weird", "datGet: Possible programming error in type conversion",
               status );
      }
    }
  }

 CLEANUP:

  if (*status != SAI__OK) {
    emsRepf("datGet_N", "datGet: Error reading data from primitive object %s as type %s"
            " (internally type is %s)",
            status, namestr, normtypestr, datatypestr);
  }

  if (tmpvalues) MEM_FREE(tmpvalues);
  if (h5type) H5Tclose(h5type);
  if (mem_dataspace_id > 0) H5Sclose(mem_dataspace_id);
  return *status;

}
