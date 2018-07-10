/*
*+
*  Name:
*     datParen

*  Purpose:
*     Locate parent structure

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datParen( const HDSLoc * locator1, HDSLoc ** locator2, int * status );

*  Arguments:
*     locator1 = const HDSLoc * (Given)
*        Object locator.
*     locator2 = HDSLoc ** (Returned)
*        Parent structure locator
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     The routine returns a locator for the parent structure of an HDS object; i.e.
*     the structure which contains the object.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     -  On successful exit, the parent structure locator will identify
*     a scalar structure (number of dimensions zero). If appropriate,
*     this may be a cell of a structure array.
*     -  An error will result, and the STATUS value DAT__OBJIN will be
*     returned if the object supplied does not have a parent; i.e. if
*     it is the top-level object in a container file. The DAT__OBJIN
*     error code is defined in the include file DAT_ERR.
*     -  If this routine is called with STATUS set, then a value of
*     DAT__NOLOC (Fortran) / NULL (C) will be returned for the LOC2 argument,
*     although no further processing will occur. The same value will also be
*     returned if the routine should fail for any reason. The
*     DAT__NOLOC constant is defined in the Fortran include file DAT_PAR.
*     - If the input locator is a member of a group the parent locator
*     will also be a member of the group.

*  Example:
*     The parent of the object A.B.C.D is A.B.C, the parent of
*     X.DATA.ARRAY(1:256) is X.DATA, and the parent of Z.STRUC(17).FLAG
*     is Z.STRUC(17).

*  History:
*     2014-09-03 (TIMJ):
*        Initial version
*     2014-11-14 (TIMJ):
*        Child locators must inherit group
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
datParen( const HDSLoc *locator1, HDSLoc **locator2, int *status ) {
  hid_t objid = 0;
  HDSLoc * thisloc = NULL;
  hid_t parent_id = 0;

  *locator2 = NULL;
  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datParen", 1, locator1, 1, status );

  /* Need to get the relevant identfier */
  objid = dat1RetrieveIdentifier( locator1, status );

  /* Get the parent group. Do not want the root group */
  parent_id = dat1GetParentID( objid, 1, status );

  thisloc = dat1AllocLoc( status );

  if (*status == SAI__OK) {
    thisloc->handle = locator1->handle->parent;
    thisloc->group_id = parent_id;
    thisloc->isprimary = HDS_FALSE;
    thisloc->file_id = locator1->file_id;
    hds1RegLocator( thisloc, status );
    if ( (locator1->grpname)[0] != '\0') hdsLink(thisloc, locator1->grpname, status);
  }

  if (*status != SAI__OK) {
    datAnnul( &thisloc, status );
  } else {
    *locator2 = thisloc;
  }
  return *status;
}
