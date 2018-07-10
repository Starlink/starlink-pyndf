/*
*+
*  Name:
*     datClone

*  Purpose:
*     Clone locator

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datClone(const HDSLoc *locator1, HDSLoc **locator2, int *status);

*  Arguments:
*     locator1 = const HDSLoc * (Given)
*        Source locator.
*     locator2 = HDSLoc ** (Returned)
*        Cloned locator.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Clone (duplicate) a locator. This locator can be used independently and
*     must be annulled explicitly.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - All information is inherited except that concerned with any
*     mapped primitive data and the primary/secondary locator
*     characteristic (a secondary locator is always produced - see
*     datPrmry). A call to this routine is NOT equivalent to the
*     assignment statement "LOC2 = LOC1".

*  History:
*     2014-09-04 (TIMJ):
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

#include <string.h>

#include "hdf5.h"

#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

int
datClone(const HDSLoc *locator1, HDSLoc **locator2, int *status) {

  HDSLoc * clonedloc = NULL;

  *locator2 = NULL;
  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datClone", 1, locator1, 1, status );

  clonedloc = dat1AllocLoc( status );
  if (*status != SAI__OK) goto CLEANUP;

  /* Lose primary-ness. */
  clonedloc->isprimary = HDS_FALSE;
  clonedloc->file_id = locator1->file_id;
  hds1RegLocator( clonedloc, status );

  if (locator1->dataset_id > 0) {
    CALLHDF( clonedloc->dataset_id,
             H5Dopen2( locator1->dataset_id, ".", H5P_DEFAULT ),
             DAT__HDF5E,
             emsRep("datClone_1", "Error opening a dataset during clone",
                    status )
             );
  }
  if (locator1->dataspace_id > 0) {
    CALLHDF( clonedloc->dataspace_id,
             H5Scopy( locator1->dataspace_id ),
             DAT__HDF5E,
             emsRep("datClone_2", "Error copying a dataspace during clone",
                    status )
             );
  }
  if (locator1->group_id > 0) {
    CALLHDF( clonedloc->group_id,
             H5Gopen2( locator1->group_id, ".", H5P_DEFAULT ),
             DAT__HDF5E,
             emsRep("datClone_3", "Error opening a group ID during clone",
                    status )
             );
  }

  /* Retain knowledge of vectorization, slicing, etc */
  clonedloc->vectorized = locator1->vectorized;
  clonedloc->isslice = locator1->isslice;
  clonedloc->iscell = locator1->iscell;
  clonedloc->isdiscont = locator1->isdiscont;
  clonedloc->handle = locator1->handle;

 CLEANUP:
  if (*status != SAI__OK) {
    if (clonedloc) datAnnul( &clonedloc, status );
  } else {
    *locator2 = clonedloc;
  }
  return *status;

}
