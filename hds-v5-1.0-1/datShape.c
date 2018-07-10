/*
*+
*  Name:
*     datShape

*  Purpose:
*     Enquire object shape

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datShape( const HDSLoc *locator, int maxdim, hdsdim dims[],
*               int *actdim, int * status );

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Object locator
*     maxdim = int (Given)
*        Allocated size of dims[]
*     dims = hdsdim [] (Returned)
*        Object dimensions.
*     actdim = int * (Returned)
*        Number of dimensions filled in dims[].
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Enquire the shape of an object.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-08-29 (TIMJ):
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

#include "hdf5.h"

#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

int
datShape( const HDSLoc *locator, int maxdim, hdsdim dims[],
          int *actdim, int * status ) {

  int i = 0;
  int rank = 0;
  hdsdim upper[DAT__MXDIM];
  hdsdim lower[DAT__MXDIM];
  hdsbool_t issubset = 0;

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datShape", 1, locator, 1, status );

  /* Single cells should always be considered scalar. */
  if( locator->iscell ){
    for (i=0; i<maxdim; i++) {
      dims[i] = 0;
    }
    rank = 0;

  /* Otherwise return the full shape. */
  } else {
    dat1GetBounds( locator, lower, upper, &issubset, &rank, status );

    if (rank > maxdim) {
      *status = DAT__DIMIN;
      emsRepf("datshape_1b", "datShape: Dimensions of object exceed maximum allowed size of %d",
              status, maxdim);
      goto CLEANUP;
    }

    /* Convert bounds to dims */
    for (i=0; i<rank; i++) {
      dims[i] = upper[i] - lower[i] + 1;
    }

    /* If a scalar is vectorised, it becomes a 1-element vector. */
    if( rank == 0 && locator->vectorized ) {
       rank = 1;
       dims[0] = 1;
    }
  }

  *actdim = rank;

 CLEANUP:
  return *status;
}
