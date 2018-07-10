/*
*+
*  Name:
*     dat1RetrieveContainer

*  Purpose:
*     Retrieve the containing structure or file ID from HDS locator

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hid_t dat1RetrieveContainer( const HDSLoc * locator, int * status );

*  Arguments:
*     locator = const HDSLoc * (Given)
*        HDS locator containing relevant items
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Look through the supplied locator and return either the relevant group
*     identifier or file identifier. If a primitive type is present in the locator
*     (even if associated with a file identifier) then this is an error.

*  Returned Value:
*     container = hid_t
*        Either file or group identifier.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - group_id is returned in preference. file_id is only returned if no primitive
*       type is present.

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

#include "hds1.h"
#include "dat1.h"
#include "ems.h"

#include "dat_err.h"
#include "sae_par.h"

hid_t dat1RetrieveContainer( const HDSLoc * locator, int * status ) {

  if (*status != SAI__OK) return 0;

  /* Group should always be returned in preference */
  if (locator->group_id > 0) return locator->group_id;

  /* if this seems to correspond to a dataset locator then this is bad */
  if (locator->dataset_id > 0) {
    *status = DAT__TYPIN;
    emsRep("dat1RetrieveContainer_1",
           "Supplied locator corresponds to primitive type and not group or root",
           status);
    return 0;
  }

  if (locator->file_id > 0) return locator->file_id;

  /* Something wrong */
  *status = DAT__TYPIN;
  emsRep("dat1RetrieveContainer_2",
         "Supplied locator does not correspond to structure or root file",
         status);
  return 0;

}
