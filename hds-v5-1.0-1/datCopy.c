/*
*+
*  Name:
*     datCopy

*  Purpose:
*     Copy object

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datCopy( const HDSLoc *locator1, const HDSLoc *locator2,
*              const char *name, int *status);

*  Arguments:
*     locator1 = const HDSLoc * (Given)
*        Locator of object to copy.
*     locator2 = const HDSLoc * (Given)
*        Locator of structure to receive the copy.
*     name = const char * (Given)
*        Name of newly copied object.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Recursively copy an object into a component. This means that the
*     complete object (including its components and its components's
*     components, etc.) is copied, not just the top level.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Uses H5Ocopy to do a deep copy. This is not a hard link.

*  History:
*     2014-09-04 (TIMJ):
*        Initial version
*     2014-11-22 (TIMJ):
*        Understand the possibility that we are copying the root group
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
#include "star/util.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

int
datCopy( const HDSLoc *locator1, const HDSLoc *locator2,
         const char *name_str, int *status) {

  char sourcename[DAT__SZNAM+1];
  char cleanname[DAT__SZNAM+1];
  hid_t parent_id = -1;
  hid_t objid = -1;

  if (*status != SAI__OK) return *status;

  /* Validate input locators. */
  dat1ValidateLocator( "datCopy", 1, locator1, 1, status );
  dat1ValidateLocator( "datCopy", 1, locator2, 0, status );

  dau1CheckName( name_str, 1, cleanname, sizeof(cleanname), status );
  if (*status != SAI__OK) return *status;

  /* Have to give the source name as "." doesn't seem to be allowed.
     so get the name and the parent locator. */
  objid = dat1RetrieveIdentifier( locator1, status );

  /* If we are at the root group we can not get a parent so just use
     the "/" name instead */
  parent_id = dat1GetParentID( objid, 1, status );
  if (*status == DAT__OBJIN) {
    emsAnnul(status);
    star_strlcpy( sourcename, "/", sizeof(sourcename));
    parent_id = -1;
  } else {
    datName( locator1, sourcename, status );
  }
  CALLHDFQ(H5Ocopy( (parent_id == -1 ? objid : parent_id), sourcename,
                    locator2->group_id, cleanname, H5P_DEFAULT, H5P_DEFAULT));

 CLEANUP:
  if (parent_id) H5Gclose(parent_id);
  return *status;

}
