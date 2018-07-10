/*
*+
*  Name:
*     dat1AllocLoc

*  Purpose:
*     Allocate memory for an HDS locator

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     HDSLoc * dat1AllocLoc( int * status );

*  Arguments:
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Returned Value:
*     loc = HDSLoc *
*        Newly allocated (empty) locator.

*  Description:
*     Allocate memory for a locator but do not fill it.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Locator must be freed by calling dat1FreeLoc()

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

#include <stdlib.h>

#include "ems.h"

#include "hds1.h"
#include "dat1.h"

#include "dat_err.h"
#include "sae_par.h"

HDSLoc *
dat1AllocLoc( int * status ) {
  HDSLoc * newloc;
  if (*status != SAI__OK) return NULL;

  newloc = MEM_CALLOC( 1, sizeof(*newloc) );

  if (!newloc) {
    *status = DAT__NOMEM;
    emsRep("dat1AllocLoc", "Could not allocate memory for HDS locator",
           status );
  }
  /* Force the implementation version into the struct */
  newloc->hds_version = 5;
  return newloc;
}
