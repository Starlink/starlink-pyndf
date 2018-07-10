/*
*+
*  Name:
*     datState

*  Purpose:
*     Enquire object state

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datState( const HDSLoc *locator, hdsbool_t *state, int *status);

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Primitive locator.
*     state = hdsbool_t * (Returned)
*        1 if defined, otherwise 0.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Enquire the state of a primitive, ie. whether its value is defined or not.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-10-16 (TIMJ):
*        Initial version
*     2014-11-21 (TIMJ):
*        If the attribute is missing, query the dataset directly.
*     2014-11-21 (TIMJ):
*        Remove attribute. Just query dataset.
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
datState( const HDSLoc *locator, hdsbool_t *state, int *status) {
  H5D_space_status_t dstatus = 0;
  *state = HDS_FALSE;

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datState", 1, locator, 1, status );

  if (dat1IsStructure(locator, status)) {
    *status = DAT__OBJIN;
    emsRep("datState_1", "datState can only be called on primitive locator",
           status);
    return *status;
  }

  /* Query the dataset to determine whether it has been allocated yet */
  CALLHDFQ( H5Dget_space_status( locator->dataset_id, &dstatus) );
  *state = ( (dstatus == H5D_SPACE_STATUS_ALLOCATED ||
              dstatus == H5D_SPACE_STATUS_PART_ALLOCATED)
             ? HDS_TRUE : HDS_FALSE );

 CLEANUP:
  return *status;
}
