/*
*+
*  Name:
*     dat1GetDataDims

*  Purpose:
*     Obtain the dimensions of the full array - not just the slice.

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     dat1GetDataDims( const HDSLoc * locator, hdsdim dims[DAT__MXDIM],
*                      int *actdim, int * status );

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Locator for which dimensions are to be obtained.
*     dims = hdsdim [DAT__MXDIM] (Returned)
*        On exit, contains the required dimensions.
*     actdim = int * (Returned)
*        Number of dimensions in array.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Obtain the dimensions of the full array associated with the
*     supplied locator, in HDS coordinate order. If the locator
*     represents a slice of an array, the dimensions of the full array -
*     not just the slice - are returned. If the locator has been
*     vectorised using datVec, the total number of elements in the
*     full array will be returned as the one and only dimensions
*     ("*actdim" will be returned set to 1).

*  Authors:
*     DSB: David S Berry (EAO)
*     {enter_new_authors_here}

*  History:
*     2017-06-14 (DSB):
*        Initial version
*     {enter_further_changes_here}

*  Copyright:
*     Copyright (C) 2017 East Asian Observatory.
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
dat1GetDataDims( const HDSLoc * locator, hdsdim dims[DAT__MXDIM],
                 int *actdim, int * status ){
  int rank = 0;
  hsize_t h5dims[DAT__MXDIM];

  /* Initialise returned values */
  *actdim = 0;

  /* Check inherited status */
  if (*status != SAI__OK) return *status;

  /* First deal with structures. These cannot currently be sliced. */
  if (dat1IsStructure( locator, status ) ) {

    /* Query the dimensions of the structure. */
    rank = dat1GetStructureDims( locator, DAT__MXDIM, dims, status );

  /* Now deal with primitives. */
  } else {

    /* Get the HDF5 dimensions (i.e. the "extent") of the full dataspace
       associated with the supplied locator. This will have rank 1 if the
       locator has been vectorised by datVec. */
    CALLHDFE( int,
              rank,
              H5Sget_simple_extent_dims( locator->dataspace_id, h5dims, NULL ),
              DAT__DIMIN,
              emsRep(" ", "Error obtaining shape of object", status) );

    /* Convert the dimensions from HDF5 order to HDS order. */
    dat1ExportDims( rank, h5dims, dims, status );
  }

  /* Return the number of dimensions. */
  *actdim = rank;

 CLEANUP:
  return *status;
}
