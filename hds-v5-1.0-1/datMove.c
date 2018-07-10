/*
*+
*  Name:
*     datMove

*  Purpose:
*     Move an object

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datMove( HDSLoc **locator1, const HDSLoc *locator2, const char *name_str,
*              int *status );

*  Arguments:
*     locator1 = HDSLoc ** (Given and Returned)
*        Object locator to move. Locator will be annulled and
*        set to NULL after moving.
*     locator2 = const HDSLoc * (Given)
*        Structure to receive the item.
*     name = const char * (Given)
*        Name of component in new location.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Move and object to a new location (i.e. copy and erase the original).

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - If the object is an array, locator1 must point to the complete
*     array, not a slice or cell. locator1 is annulled if the operation is
*     successful (if it is the last primary locator associated with a
*     container file, then the container file will be closed - see
*     datPrmry but note that the HDF5 interface does not currently support
*     primary vs secondary locators).
*     - The operation will fail if a component of the same
*     name already exists in the structure. The object to be moved
*     need not be in the same container file as the structure.

*  History:
*     2014-09-04 (TIMJ):
*        Initial version
*     2014-11-04 (TIMJ):
*        H5Lmove can only move items within a file so use datCopy/datErase
*        if it seems that this is a move between files).
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
datMove( HDSLoc **locator1, const HDSLoc *locator2, const char *name_str,
         int *status ) {

  HDSLoc * parentloc = NULL;
  char sourcename[DAT__SZNAM+1];
  char cleanname[DAT__SZNAM+1];

  if (*status != SAI__OK) return *status;

  /* Validate input locators. */
  dat1ValidateLocator( "datMove", 1, *locator1, 0, status );
  dat1ValidateLocator( "datMove", 1, locator2, 0, status );

  dau1CheckName( name_str, 1, cleanname, sizeof(cleanname), status );
  if (*status != SAI__OK) return *status;

  /* Have to give the source name as "." doesn't seem to be allowed.
     so get the name and the parent locator */
  datParen( *locator1, &parentloc, status );
  datName( *locator1, sourcename, status );

  /* H5Lmove can only move within a file. If we are moving
     between files we need to do this manually with datCopy/datErase.
     At the moment not clear how to see if the file is the same so just
     compare file_id.
  */
  if ((*locator1)->file_id == locator2->file_id) {
    CALLHDFQ(H5Lmove( parentloc->group_id, sourcename,
                      locator2->group_id, cleanname, H5P_DEFAULT, H5P_DEFAULT));
  } else {
    datCopy( *locator1, locator2, name_str, status );
    datErase( parentloc, sourcename, status );
  }

 CLEANUP:
  if (parentloc) datAnnul( &parentloc, status );
  if (*status == SAI__OK) {
    datAnnul(locator1, status );
  }
  return *status;
}
