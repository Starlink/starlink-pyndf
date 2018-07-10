/*
*+
*  Name:
*     dat1Index2Coords

*  Purpose:
*     Convert array index to array coordinates

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     dat1Index2Coords ( size_t idx, int ndim, const hdsdim arraydims[DAT__MXDIM],
*                        hdsdim coords[DAT__MXDIM], int *status );

*  Arguments:
*     idx = size_t (Given)
*        Index into array of dimension "arraydims".
*     ndim = int (Given)
*        Number of dimensions in "arraydims".
*     arraydims = const hdsdim [DAT__MXDIM]
*        Dimensions (1-based, HDS order) of the array being indexed.
*     coords = hdsdim [DAT__MXDIM]
*        1-based coordinates within "arraydims" of the supplied index "idx".
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Given a 1-based index, and the dimensions of the N-D array,
*     return the 1-based N-D coordinates. For example, in a 3-D
*     array of dimensions (4,3,2), index 19 is element (3,2,2)
*     and index 6 is (2,2,1).

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-09-12 (TIMJ):
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

void
dat1Index2Coords ( size_t idx, int ndim, const hdsdim arraydims[DAT__MXDIM],
                   hdsdim coords[DAT__MXDIM], int *status ) {

  int curdim;
  int prevdim;

  if (*status != SAI__OK) return;

  /*
    Loop over one fewer dimensions than we actually have
    subtracting the biggest dimensions from idx as we go.
    The final coordinate value is simply the remainder
    when we finish looping
  */

  for (curdim = 1; curdim < ndim; curdim++) {
    size_t intdiv;
    size_t elems_prior = 1;
    /* Calculate how many elements are covered by full
       earlier dimensions */
    for (prevdim = 1; prevdim <= (ndim-curdim); prevdim++) {
      elems_prior *= arraydims[prevdim-1]; /* zero based lookup */
    }

    /* Calculate the coordinate for the current dim by dividing
       by the number of elements prior using integer division. Need to
       subtract one from the result for 1-based counting. */
    intdiv = (idx-1) / elems_prior;

    /* Store the coordinate, starting from the end. The +1 is
       for 1-based counting. */
    coords[ndim-curdim] = intdiv + 1;

    /* And subtract all those elements from the supplied index and go
       round again */
    idx -= intdiv * elems_prior;

  }

  /* The final value for idx is the final coordinate value */
  coords[0] = idx;

}
