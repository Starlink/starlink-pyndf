/*
*+
*  Name:
*     datSize

*  Purpose:
*     Enquire object size

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datSize(const HDSLoc *locator, size_t *size,  int * status );

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Object locator
*     size = size_t * (Returned)
*        Object size
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Enquire the size of an object. For an array this will be the product of the
*     dimensions; for a scalar, a value of 1 is returned.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-08-26 (TIMJ):
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

#include "star/one.h"
#include "ems.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"
#include "sae_par.h"

int
datSize(const HDSLoc *locator,
        size_t *size,
        int *status ) {

  hdsdim objdims[DAT__MXDIM];
  int ndims = 0;
  int i = 0;
  size_t nelem = 1;

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datSize", 1, locator, 1, status );

  /* Do not duplicate code from datShape -- just call it and we know then
     that it works for arrays of structures and for slices */
  datShape( locator, DAT__MXDIM, objdims, &ndims, status );

  if (*status == SAI__OK) {
    for (i=0; i<ndims; i++) {
      nelem *= objdims[i];
    }
  }
  *size = nelem;
  return *status;
}
