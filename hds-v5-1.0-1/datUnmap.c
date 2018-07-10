/*
*+
*  Name:
*     datUnmap

*  Purpose:
*     Unmap object

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datUnmap( HDSLoc * locator, int * status );

*  Arguments:
*     locator = HDSLoc * (Given and Returned)
*        Primitive locator. Previously mapped with datMap or related
*        routine.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Unmap an object mapped by another datX routine.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - If status is bad on entry an attempt will be made to sync
*       data back to the file and resources freed, but the sync is
*       not guaranteed (depending on the reason the status was bad).
*     - API differs slightly from HDS in that the supplied
*       locator can not be const as its state is updated.

*  History:
*     2014-08-29 (TIMJ):
*        Initial version
*     2014-11-06 (TIMJ):
*        If pointer was memory mapped directly from a file close
*        the file and do not use datPut.
*     2014-11-20 (TIMJ):
*        Use cnfFree if the pointer was not actually mapped.
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

#include <errno.h>
#include <sys/mman.h>
#include <unistd.h>

#include "hdf5.h"

#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"
#include "f77.h"
#include "dat_err.h"

int
datUnmap( HDSLoc * locator, int * status ) {
  /* Try to unmap even if status is bad */
  int lstat = SAI__OK;

  /* Just ignore a null pointer */
  if (!locator) return *status;

  /* if there is no mapped pointer in this locator do nothing */
  if (!locator->regpntr) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datUnmap", 1, locator, (locator->accmode & HDSMODE_READ), status );

  /* We only copy back explicitly if we did not do a native mmap on the file */
  if (!locator->uses_true_mmap) {

    /* If these data were mapped for WRITE or UPDATE we have to copy
       back the data. Use datPut() for that. Mark the error stack
       before we try this.
    */

    emsMark();

    if (locator->accmode == HDSMODE_WRITE ||
        locator->accmode == HDSMODE_UPDATE) {
      datPut( locator, locator->maptype, locator->ndims, locator->mapdims,
              locator->regpntr, &lstat);
    }

    /* if we have bad status from this just ignore it. Release the error stack */
    if (lstat != SAI__OK) emsAnnul( &lstat );
    emsRlse();
  }

  /* Need to free the memory and, if needed, unregister the pointer.
     If "pntr" is defined then this was mmapped. */
  if (locator->pntr) {
    cnfUregp( locator->regpntr );

    if ( munmap( locator->pntr, locator->bytesmapped ) != 0 ) {
      if (*status == SAI__OK) {
        *status = DAT__FILMP;
        emsSyser( "MESSAGE", errno );
        emsRep("datUnMap_4", "datUnmap: Error unmapping mapped memory: ^MESSAGE", status);
      }
    }
  } else if (locator->regpntr) {
    /* Allocated memory that needs to be freed by CNF but was not mmapped */
    cnfFree( locator->regpntr );
  }

  /* If these data were mmap-ed directly on disk in WRITE mode then
     we cause an error as this has not been tested. */
  if (locator->uses_true_mmap) {
    if (locator->accmode == HDSMODE_WRITE) {
      if (*status == SAI__OK) {
        *status = DAT__FATAL;
        emsRep("datUnmap_no", "datUnmap: Unexpectedly mapped an array in WRITE mode",
               status);
      }
    }
  }

  locator->pntr = NULL;
  locator->regpntr = NULL;
  locator->bytesmapped = 0;

  /* Close the file if we opened it -- ignore the return value */
  if (locator->fdmap > 0) {
    close(locator->fdmap);
    locator->fdmap = 0;
  }

  return *status;
}
