/*
*+
*  Name:
*     datPrmry

*  Purpose:
*     Set or enquire primary/secondary locator status

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datPrmry(int set, HDSLoc **locator, hdsbool_t *prmry, int *status);

*  Arguments:
*     set = hdsbool_t (Given)
*        If a true value is given for this argument, then the routine will perform a "set"
*        operation to set the primary/secondary status of a locator. Otherwise it will perform
*        an "enquire" operation to return the value of this status without changing it.
*     locator = HDSLoc ** (Given and Returned)
*        The locator whose primary/secondary status is to be set or enquired. Will be
*        set to NULL if the locator is the last primary locator and it is set to
*        secondary status.
*     prmry = hdsbool_t * (Given and Returned)
*        If "set" is true, then this is an input argument and specifies the new value to
*        be set (true for a primary locator, false for a secondary locator). If "set" is
*        false, then this is an output argument and will return a value indicating whether
*        or not a primary locator was supplied.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Can be used to set or query the primary or secondary status of
*     a locator. For each open file (hdsOpen or hdsNew) the locators
*     are tracked and if the number of primary locators associated with
*     the file drops to zero all the locators associated with a file
*     will be annulled.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  See Also:
*     - datRefct to determine whether changing the supplied locator
*       to a secondary locator will cause the file to be closed.

*  Notes:
*     - Primary status is stored internally in the locator and
*       tracked in a per-file data structure.
*     - The locator argument is a pointer to a pointer because
*       in theory after demoting a locator it is possible that this
*       would result in the file closing and the locator being
*       annulled.
*     - Locators from hdsOpen and hdsNew are always primary. Locators
*       from datFind and datSlice are secondary locators.

*  History:
*     2014-09-09 (TIMJ):
*        Initial version
*     2014-10-30 (TIMJ):
*        Now basic system implemented although it might not
*        behave in the same way as HDSv4 behaved.
*     2014-11-10 (TIMJ):
*        Use central registry of locators to implement primary/secondary
*        in a way that matched HDSv4.
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
datPrmry(hdsbool_t set, HDSLoc **locator, hdsbool_t *prmry, int *status) {

  if (*status != SAI__OK) return *status;

  if (set) {
    if (*prmry) {
      (*locator)->isprimary = HDS_TRUE;
    } else {
      /* check if we need to do something */
      if ( (*locator)->isprimary ) {
        int refct = 0;
        /* We are converting a primary to a secondary locator.
           Get the reference count for this file id */
        datRefct( *locator, &refct, status );

        if (refct == 1) {
          /* This locator is primary and there is only one
             primary associated with this file id. That means
             the file should be closed and locator freed. */
          hds1FlushFile( (*locator)->file_id, status );
          *locator = NULL;
        } else {
          (*locator)->isprimary = HDS_FALSE;
        }
      }
    }

  } else {
    *prmry = (*locator)->isprimary;
  }
  return *status;
}
