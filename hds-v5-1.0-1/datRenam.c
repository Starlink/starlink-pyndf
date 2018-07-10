/*
*+
*  Name:
*     datRenam

*  Purpose:
*     Rename object

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datRenam( HDSLoc *locator, const char *name_str, int  *status);

*  Arguments:
*     locator = HDSLoc * (Given and Returned)
*        Object locator. Updated to reflect new name of object.
*     name_str = const char * (Given)
*        New object name.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Rename an object in the hierarchy.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - This locator is updated by the routine to reflect the new location
*       of the object in the hierarchy. This is a change in API compared to
*       HDS.

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

#include "hdf5.h"

#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

int
datRenam( HDSLoc *locator, const char *name_str, int  *status) {

  HDSLoc * clonedloc = NULL;
  HDSLoc * parentloc = NULL;
  HDSLoc * movedloc = NULL;
  int there = 0;

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datRenam", 1, locator, 0, status );

  /* We need to get the locator of the parent as we need to be moving it
     within the current group. */
  datParen( locator, &parentloc, status );

  /* It is an error to rename over something that already exists */
  datThere( parentloc, name_str, &there, status );
  if (there) {
    *status = DAT__COMEX;
    emsRepf("datRenam_1", "datRenam: Component '%s' already exists in this structure",
            status, name_str );
    goto CLEANUP;
  }

  /* To rename we are actually moving. HDF5 does not see any difference
     between moving and renaming so we just call datMove. */

  /* First we clone the input locator so that we do not free the caller's locator. */
  datClone( locator, &clonedloc, status );

  /* Move the item to the new location */
  datMove( &clonedloc, parentloc, name_str, status );

  /* but now we have to find the thing we just moved */
  datFind( parentloc, name_str, &movedloc, status );

  /* and we now do the fiddly bit where we have to
     replace the bits in the callers locator */
#define COPYCOMP(item,closefunc)                                        \
  if (*status == SAI__OK) {                                             \
    if (movedloc->item > 0 && locator->item > 0) {                      \
      CALLHDFQ(closefunc(locator->item));                               \
      locator->item = movedloc->item;                                   \
      movedloc->item = 0; /* Null it out */                             \
    } else if (movedloc->item <= 0 && locator->item > 0) {              \
      *status = DAT__OBJIN;                                             \
      emsRep("datRenam_2", "Original locator has " #item                \
             " but renamed locator does not (Possible programming error)", \
             status );                                                  \
    } else if (movedloc->item > 0 && locator->item <= 0) {              \
      *status = DAT__OBJIN;                                             \
      emsRep("datRenam_2", "Renamed locator has " #item                 \
             " but original locator does not (Possible programming error)", \
             status );                                                  \
    }                                                                   \
  }

  COPYCOMP( dataset_id, H5Dclose );
  COPYCOMP( group_id, H5Gclose );
  COPYCOMP( dataspace_id, H5Sclose );

 CLEANUP:
  if (clonedloc) datAnnul( &clonedloc, status );
  if (parentloc) datAnnul( &parentloc, status );
  if (movedloc) datAnnul( &movedloc, status );

  return *status;
}
