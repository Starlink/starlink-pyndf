/*
*+
*  Name:
*     dat1ExportDims

*  Purpose:
*     Export dimensions from HDF5 to HDS form

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     void dat1ExportDims( int ndims, const hsize_t h5dims[], hdsdim hdsdims[],
*                          int * status );

*  Arguments:
*     ndims = int (Given)
*        Number of dimensions to export.
*     h5dims = const hsize_t [] (Given)
*        HDF5 Dimensions to import. Only ndims will be accessed.
*     hdsdims = hdsdim[] (Returned)
*        Array to receive imported dimensions. Must be at least ndims in size.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Export the dimensions in HDF5 form and convert them to the dimensions
*     suitable for use in caller code that is assuming HDS compatibility.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Does not assume that hdsdim and hsize_t are the same type.
*     - Transposes the order as HDS uses Fortran order and HDF5 uses
*       C order.

*  History:
*     2014-09-03 (TIMJ):
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
dat1ExportDims( int ndims, const hsize_t h5dims[], hdsdim hdsdims[],
                int *status ) {
  int i;

  if (*status != SAI__OK) return;
  if (ndims == 0) return;

  /* We may have to transpose these dimensions */
  for (i=0; i<ndims; i++) {
    int oposn = ndims - 1 - i;
    hdsdims[oposn] = h5dims[i];
  }
  return;
}
