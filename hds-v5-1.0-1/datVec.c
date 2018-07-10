/*
*+
*  Name:
*     datVec

*  Purpose:
*     Vectorise object

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datVec( const HDSLoc *locator1, HDSLoc **locator2, int *status );

*  Arguments:
*     locator1 = const HDSLoc * (Given)
*        Array locator
*     locator2 = HDSLoc ** (Returned)
*        Vector locator
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Address an array as if it were a vector.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     DSB: David S Berry (EAO)
*     {enter_new_authors_here}

*  Notes:
*     - It is an error to vectorize a sliced locator.

*  History:
*     2014-09-08 (TIMJ):
*        Initial version
*     2014-11-13 (TIMJ):
*        Create a 1D dataspace
*     2017-06-14 (DSB):
*        Re-written to allow contiguous slices to be vectorised. The
*        new approach is to vectorise the dataspace in the locator
*        structure.
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
datVec( const HDSLoc *locator1, HDSLoc **locator2, int *status ) {

  hdsbool_t issubset;
  hdsdim dims[DAT__MXDIM];
  hdsdim lower[DAT__MXDIM];
  hdsdim upper[DAT__MXDIM];
  hsize_t block;
  hsize_t count;
  hsize_t h5max;
  hsize_t *maxptr;
  hsize_t ifirst;
  hsize_t ilast;
  hsize_t plane_size;
  int actdims;
  int i;

  /* Initialise returned values. */
  *locator2 = NULL;

  /* Check inherited status. */
  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datVec", 1, locator1, 1, status );

  /* We cannot vectorise a discontiguous slice of an array. */
  if( *status == SAI__OK && locator1->isdiscont ) {
    *status = DAT__OBJIN;
    emsRep("datVec_1", "datVec: Cannot vectorise a discontiguous slice",
           status );
  }

  /* Create a new locator by cloning the supplied locator. */
  datClone(locator1, locator2, status);

  /* Get the dimensions and rank of the full array - not just any slice
     represented by the supplied locator. If it is already one-dimensional,
     or if it has already been vectorised, there is nothing more to do. */
  dat1GetDataDims( locator1, dims, &actdims, status );
  if( *status == SAI__OK && actdims != 1 && !locator1->vectorized ) {

    /* The supplied locator may represent a slice of the full array. Find
       the lower and upper bounds within the full array, of the supplied
       locator. */
    dat1GetBounds( locator1, lower, upper, &issubset, &actdims, status );

    /* Find the zero-based 1-dimensional index within the full array, at the
       first and last pixel of the supplied locator. Also get the total
       number of pixels in the full array. Remember that the HDS bounds
       are 1-based. */
    ifirst = 0;
    ilast = 0;
    plane_size = 1;
    for( i = 0; i < actdims; i++ ) {
      ifirst += ( lower[ i ] - 1 )*plane_size;
      ilast += ( upper[ i ] - 1 )*plane_size;
      plane_size *= dims[ i ];
    }

#if HDS_USE_CHUNKED_DATASETS
    h5max = H5S_UNLIMITED;
    maxptr = &h5max;
#else
    maxptr = NULL;
#endif


    if( (*locator2)->dataspace_id ) {
      H5Sclose((*locator2)->dataspace_id );
      (*locator2)->dataspace_id = 0;
    }

    /* If the locator has an existing dataspace, modify its extent to
       represent a 1-dimensional (i.e. vectorised) version of the full
       array. */
    if( (*locator2)->dataspace_id ) {
      CALLHDFQ( H5Sselect_none( (*locator2)->dataspace_id ) );
      CALLHDFQ( H5Sset_extent_simple( (*locator2)->dataspace_id, 1,
                                       &plane_size, maxptr ) );

    /* If the locator has no existing dataspace, create a new one
       representing a 1-dimensional (i.e. vectorised) version of the full
       array. */
    } else {
      CALLHDF( (*locator2)->dataspace_id,
               H5Screate_simple( 1, &plane_size, maxptr ),
               DAT__HDF5E,
               emsRepf(" ", "Error allocating data space", status )
             );
    }

    /* If the locator does not represent the full array, select a 1D
       hyperslab of the vectorised dataset that contains the same pixels
       as the supplied locator. */
    count = 1;
    block = ilast - ifirst + 1;
    if( count < plane_size ) {
      CALLHDFQ( H5Sselect_hyperslab( (*locator2)->dataspace_id, H5S_SELECT_SET,
                                     &ifirst, NULL, &count, &block ) );
    }

    /* Indicate the object has been vectorised, and so cannot be a scalar cell. */
    (*locator2)->vectorized = 1;
    (*locator2)->iscell = 0;
  }

 CLEANUP:
  if (*status != SAI__OK) {
    if (*locator2 != NULL) datAnnul(locator2, status);
  }
  return *status;
}
