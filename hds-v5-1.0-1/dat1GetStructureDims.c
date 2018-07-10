/*
*+
*  Name:
*     dat1GetStructureDims

*  Purpose:
*     Obtains dimensions of a structure

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     rank = dat1GetStructureDims( const HDSLoc * locator, int maxdims, hdsdim dims[],
*                                  int *status );

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Locator to structure object.
*     maxdims = int (Given)
*        Allocated size of dims[]
*     dims = hdsdim (Given & Returned)
*        On exit will contain the dimensions of the structure. Will not
*        be touched if rank is 0. Only rank elements will be stored.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Calculates the dimensionality or rank of a structure and returns
*     the appropriate dimensions.

*  Returned Value:
*     rank = int
*        Dimensionality of the structure. Can be 0 for a scalar structure.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     

*  History:
*     2014-11-21 (TIMJ):
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

int
dat1GetStructureDims( const HDSLoc * locator, int maxdims, hdsdim dims[], int *status ) {

  size_t actdims = 0;
  if (*status != SAI__OK) return actdims;
  if (!H5Aexists(locator->group_id, HDS__ATTR_STRUCT_DIMS)) return actdims;

  dat1GetAttrHdsdims( locator->group_id, HDS__ATTR_STRUCT_DIMS, HDS_FALSE,
                      0, NULL, maxdims, dims, &actdims, status );
  return actdims;
}
