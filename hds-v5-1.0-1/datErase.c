/*
*+
*  Name:
*     datErase

*  Purpose:
*     Erase component.

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datErase(const HDSLoc *locator, const char *name_str, int *status);

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Structure locator.
*     name_str = const char * (Given)
*        Name of component to be erased.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Recursively delete a component. This means that all its lower level components
*     are deleted as well.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-09-20 (TIMJ):
*        Initial version
*     2014-11-13 (TIMJ):
*        Must normalize the name. Also add better error checking and reporting.
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
datErase(const HDSLoc   *locator, const char *name_str, int *status) {
  char groupstr[DAT__SZNAM+1];
  char cleanname[DAT__SZNAM+1];

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datErase", 1, locator, 0, status );

  /* containing locator must refer to a group */
  if (locator->group_id <= 0) {
    *status = DAT__OBJIN;
    emsRep("datErase_1", "Input object is not a structure",
           status);
    return *status;
  }

  /* Parent group for error reporting */
  datName( locator, groupstr, status);

  /* Ensure the name is cleaned up before we use it */
  dau1CheckName( name_str, 1, cleanname, sizeof(cleanname), status );

  CALLHDFQ( H5Ldelete( locator->group_id, cleanname, H5P_DEFAULT ));

  /* Remove the handle for the erased component and all sub-components */
  dat1EraseHandle( locator->handle, cleanname, status );

 CLEANUP:
  if (*status != SAI__OK) {
    emsRepf("datErase_2", "Error deleting component %s in group %s",
            status, name_str, groupstr);
  }
  return *status;
}
