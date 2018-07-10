/*
*+
*  Name:
*     dat1GetBounds

*  Purpose:
*     Obtain lower and upper bounds of object

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     dat1GetBounds( const HDSLoc * locator, hdsdim lower[DAT__MXDIM],
*                    hdsdim upper[DAT__MXDIM], hdsbool_t * issubset,
*                    int *actdim, int * status );

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Locator for which bounds are to be obtained.
*     lower = hdsdim [DAT__MXDIM] (Returned)
*        On exit, contains the lower bounds.
*     upper = hdsdim [DAT__MXDIM] (Returned)
*        On exit, contains the upper bounds.
*     issubset = hdsbool_t * (Returned)
*        True if the bounds refer to a subset of the full extent. False
*        if they cover the full extent.
*     actdim = int * (Returned)
*        Number of active dimensions in object.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Obtain the lower and upper bounds of the object
*     in HDS coordinate order. Can support cells and slices.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-09-15 (TIMJ):
*        Initial version
*     2014-11-21 (TIMJ):
*        Use dat1GetStructDims
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
dat1GetBounds( const HDSLoc * locator, hdsdim lower[DAT__MXDIM],
               hdsdim upper[DAT__MXDIM], hdsbool_t * issubset,
               int *actdim, int * status ) {
  int rank = 0;
  hssize_t nblocks = 0;
  hsize_t *blockbuf = NULL;

  *actdim = 0;
  *issubset = 0;
  if (*status != SAI__OK) return *status;



   /* If the supplied locator has a dataspace, then use the bounds of the
      data space. This is done even if the object is a structure, since
      vectorised structure arrays will have a dataspace describing their
      vectorised extent. */
  if( locator->dataspace_id ) {
    int i;
    hsize_t h5lower[DAT__MXDIM];
    hsize_t h5upper[DAT__MXDIM];
    hsize_t h5dims[DAT__MXDIM];

    CALLHDFE( int,
              rank,
              H5Sget_simple_extent_dims( locator->dataspace_id, h5dims, NULL ),
              DAT__DIMIN,
              emsRep("datshape_1", "datShape: Error obtaining shape of object",
                     status)
              );

    /* If we are using datSlice then there should be one (and only one) hyperslab
       for the dataspace and we need to handle that. Should be same dimensionality
       as above. Negative number indicates there were no hyperslabs. */
    if( H5Sget_select_type( locator->dataspace_id ) == H5S_SEL_HYPERSLABS ) {
       nblocks = H5Sget_select_hyper_nblocks( locator->dataspace_id );
    } else {
       nblocks = 0;
    }

    if (nblocks == 1) {
      herr_t h5err = 0;

      *issubset = 1;

      blockbuf = MEM_MALLOC( nblocks * rank * 2 * sizeof(*blockbuf) );

      CALLHDF( h5err,
               H5Sget_select_hyper_blocklist( locator->dataspace_id, 0, 1, blockbuf ),
               DAT__DIMIN,
               emsRep("datShape_2", "datShape: Error obtaining shape of slice", status )
               );

      /* We only go through one block. The buffer is returned in form:
         ndim start coordinates, then ndim opposite corner coordinates
         and repeats for each block (if we had more than one block).
      */
      for (i = 0; i<rank; i++) {
        hsize_t start;
        hsize_t opposite;
        start = blockbuf[i];
        opposite = blockbuf[i+rank];
        /* So update the shape to account for the slice: HDS is 1-based */
        h5lower[i] = start + 1;
        h5upper[i] = opposite + 1;
      }

    } else if (nblocks > 1) {
      if (*status == SAI__OK) {
        *status = DAT__WEIRD;
        emsRepf("datShape_2", "Unexpectedly got %zd hyperblocks from locator. Expected 1."
                " (possible programming error)", status, (ssize_t)nblocks);
        goto CLEANUP;
      }
    } else {
      /* No hyperblock */
      for (i=0; i<rank; i++) {
        h5lower[i] = 1;    /* HDS value 1-based */
        h5upper[i] = h5dims[i];
      }

    }

   dat1ExportDims( rank, h5lower, lower, status );
   dat1ExportDims( rank, h5upper, upper, status );

  /* If no dataspace ia available, and the locator is a structure
     array... */
  } else if (dat1IsStructure( locator, status ) ) {

    /* Query the dimensions of the structure */
    rank = dat1GetStructureDims( locator, DAT__MXDIM, upper, status );

    if (rank > 0) {
      int i;
      for (i=0; i<rank; i++) {
        lower[i] = 1;
      }

    }

  } else if( *status == SAI__OK ) {
    *status = DAT__WEIRD;
    emsRepf(" ", "Unexpectedly got primitive array with no dataspace "
            "(possible programming error)", status );
    goto CLEANUP;
  }

  *actdim = rank;

 CLEANUP:
  if (blockbuf) MEM_FREE( blockbuf );
  return *status;
}
