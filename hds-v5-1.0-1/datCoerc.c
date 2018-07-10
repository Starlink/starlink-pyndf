/*
*+
*  Name:
*     datCoerc

*  Purpose:
*     Coerce object shape

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datCoerc( const HDSLoc *locator1, int ndim, HDSLoc **locator2, int *status);

*  Arguments:
*     locator1 = const HDSLoc * (Given)
*        Object locator.
*     ndim = int (Given)
*        Number of dimensions in coerced object.
*     locator2 = HDSLoc ** (Returned)
*        Coerced object locator.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Temporarily coerce an object into changing its shape.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - If the number of dimensions in the object is to be increased, each
        additional dimension size is set to 1, e.g. if loc1 is
        associated with a 2-D object of shape (512,256) say, setting
        ndim to 3 transforms the dimensions to (512,256,1). Likewise,
        if the number of dimensions is to be reduced, the appropriate
        trailing dimension sizes are discarded; the routine will fail
        if any of these do not have the value 1. As with DAT_VEC, only
        the appearance of the object is changed - the original shape
        remains intact.
*     - Not yet implemented.

*  History:
*     2014-10-15 (TIMJ):
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

int datCoerc( const HDSLoc *locator1, int ndim, HDSLoc **locator2, int *status) {

  if (*status != SAI__OK) return *status;

  *status = DAT__FATAL;
  emsRep("datCoerc", "datCoerc: Not yet implemented for HDF5",
         status);

  return *status;
}
