/*
*+
*  Name:
*     datAnnul

*  Purpose:
*     Annul locator

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datAnnul( HDSLoc **locator, int * status );

*  Arguments:
*     locator = HDSLoc ** (Given and Returned)
*        Locator to free. Will be NULL on exit.
*     status = int* (Given and Returned)
*        Pointer to global status. Attempts to run even
*        if status is bad.

*  Description:
*     Free up resources associated with a locator. Cancel the
*     association between a locator and an object.  Any primitive
*     value currently mapped to the locator is automatically unmapped.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     This routine attempts to execute even if status is set on entry,
*     although no further error report will be made if it subsequently
*     fails under these circumstances. In particular, it will fail if
*     the locator supplied is not valid, but this will only be
*     reported if status is set to SAI__OK on entry.

*  History:
*     2014-08-26 (TIMJ):
*        Initial version
*     2014-08-29 (TIMJ):
*        datUnmap just in case.
*     2014-10-21 (TIMJ):
*        Try to remove locator from group if we are being asked
*        to free a locator stored in a group.
*     2014-10-30 (TIMJ):
*        Only annul the file_id if it is a primary locator.
*     2014-11-11 (TIMJ):
*        Use new error context. We need to unmap and unregister
*        even if status is bad.
*     {enter_further_changes_here}

*  Copyright:
*     Copyright (C) 2014 Cornell University
*     All Rights Reserved.

*  Licence:
*     This program is free software; you can redistribute it and/or
*     modify it under the terms of the GNU General Public License as
*     published by the Free Software Foundation; either version 3 of
*     the License, or (at your option) any later version.
*
*     This program is distributed in the hope that it will be
*     useful, but WITHOUT ANY WARRANTY; without even the implied
*     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
*     PURPOSE. See the GNU General Public License for more details.
*
*     You should have received a copy of the GNU General Public License
*     along with this program; if not, write to the Free Software
*     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
*     MA 02110-1301, USA.

*  Bugs:
*     {note_any_bugs_here}
*-
*/

#include "hdf5.h"

#include "sae_par.h"
#include "ems.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

int datAnnul( HDSLoc **locator, int * status ) {
  /* Attempts to run even if status is bad */
  int lstat = SAI__OK;

  /* Sanity check argument */
  if (!locator) return *status;
  if (! *locator) return *status;

  /* Begin an entirely new error context as we need to run this
     regardless of external errors */
  lstat = *status;
  emsBegin( &lstat );
  emsMark();

  /* Free file resources */
  dat1Annul( *locator, &lstat );

  /* Free the memory associated with this locator */
  *locator = dat1FreeLoc( *locator, &lstat );

  /* End the error context and return the final status */
  emsRlse();
  emsEnd( &lstat );
  *status = lstat;

  return *status;
}
