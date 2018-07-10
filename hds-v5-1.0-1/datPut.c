/*
*+
*  Name:
*     datPut

*  Purpose:
*     Write primitive

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     int datPut( const HDSLoc *locator, const char *type_str, int ndim, const hdsdim dims[],
*                 const void *values, int *status);

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Primitive locator.
*     type = const char * (Given)
*        Data type to be stored (not necessarily the data type of the locator).
*     ndim = int (Given)
*        Number of dimensions.
*     dims = const hdsdim[] (Given)
*        Object dimensions.
*     values = const void * (Given)
*        Object value of type "type" and dimensionality "dims".
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*       Write a primitive (type specified by a parameter).

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     DSB: David S Berry (EAO)
*     {enter_new_authors_here}

*  Notes:
*     - Character strings are given as a single character buffer and not as char **.
*       The type string indicates how many characters are expected per element
*       and the buffer is assumed to be space padded.

*  History:
*     2014-08-27 (TIMJ):
*        Initial version
*     2014-11-06 (TIMJ):
*        If a long name has been supplied make sure we do not care
*        by annulling the ONE__TRUNC error. This can happen when we
*        are working on temporary structures hidden from HDS.
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
#include "one_err.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"
#include "dat_err.h"

int
datPut( const HDSLoc *locator, const char *type_str, int ndim, const hdsdim dims[],
        const void *values, int *status) {

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
  int i;
  int isprim;
  void * tmpvalues = NULL;

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datPut", 1, locator, 0, status );

  datName(locator, namestr, status);

  /* we do not care because this must be a temporary component
     that we are trying to hide from HDS */
  if (*status == ONE__TRUNC) emsAnnul(status);

  /* Ensure that this locator is associated with a primitive type */
  if (locator->dataset_id <= 0) {
    *status = DAT__OBJIN;
    emsRepf("", "datPut: Can not put data into non-primitive location '%s'",
            status, namestr );
    return *status;
  }

  /* Ensure that we have a primitive type supplied */
  isprim = dau1CheckType( 1, type_str, &h5type, normtypestr,
                          sizeof(normtypestr), status );

  if (!isprim) {
    if (*status == SAI__OK) {
      *status = DAT__TYPIN;
      emsRepf("datPut_1", "datPut: Data type in %s must be a primitive type and not '%s'",
              status, namestr, normtypestr);
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
          emsRepf("", "datPut: Supplied dimension (%" HDS_DIM_FORMAT
                  ") on axis %d is incorrect - it should be %"
                  HDS_DIM_FORMAT ".", status, dims[i], i+1, locdims[i] );
          break;
        }
      }
    } else {
      *status = DAT__DIMIN;
      emsRepf("", "datPut: Supplied no. of axes (%d) is incorrect - it "
              "should be %d.", status, ndim, actdim );
    }
  }


  if (*status != SAI__OK) goto CLEANUP;

  /* Check data types and do conversion if required */
  outtype = dat1Type( locator, status );
  intype = dau1HdsType( h5type, status );

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
    size_t nbin = 0;
    size_t nbout = 0;
    size_t nbad = 0;
    size_t nelem = 0;
    hid_t tmptype = 0;

    /* Number of elements to convert */
    datSize( locator, &nelem, status );

    /* Number of bytes per element in the input type */
    CALLHDF(nbin,
            H5Tget_size( h5type ),
            DAT__HDF5E,
            emsRep("datPut_size", "datPut: Error obtaining size of input type",
                   status)
            );

    /* Number of bytes per element in the output type */
    datLen( locator, &nbout, status );

    /* Create a buffer to receive the converted values */
    tmpvalues = MEM_MALLOC( nelem * nbout );

    if (doconv == HDSTYPE_CHAR) {
      dat1CvtChar( nelem, intype, nbin, outtype, nbout, values,
                   tmpvalues, &nbad, status );
    } else {
      dat1CvtLogical( nelem, intype, nbin, outtype, nbout, values,
                      tmpvalues, &nbad, status );
    }
    /* The type of the things we are writing has now changed
       so we need to update that */
    if (h5type) H5Tclose(h5type);
    CALLHDF( h5type,
             H5Dget_type( locator->dataset_id ),
             DAT__HDF5E,
             emsRep("datPut_type", "datPut: Error obtaining data type of native dataset", status)
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
           emsRep("datPut_2", "Error allocating in-memory dataspace", status )
           );

  CALLHDFQ( H5Dwrite( locator->dataset_id, h5type, mem_dataspace_id,
                      locator->dataspace_id, H5P_DEFAULT,
                      (tmpvalues ? tmpvalues : values )
                      ) );

 CLEANUP:
  if (h5type) H5Tclose(h5type);
  if (mem_dataspace_id > 0) H5Sclose(mem_dataspace_id);
  if (tmpvalues) MEM_FREE(tmpvalues);
  if (*status != SAI__OK) {
    emsRepf("datPut_3", "datPut: Error writing data of type '%s' into primitive %s",
            status, normtypestr, namestr);
  }
  return *status;
}
