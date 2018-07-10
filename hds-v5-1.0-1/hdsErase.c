/*
*+
*  Name:
*     hdsErase

*  Purpose:
*     Erase container file.

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hdsErase(HDSLoc **locator, int *status);

*  Arguments:
*     locator = HDSLoc ** (Given and Returned)
*        Locator to the container file's top-level object. Will be annuled
*        on exit.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Mark a container file for deletion and annul the locator
*     associated with the top-level object. The container file will
*     not be physically deleted if other primary locators are still
*     associated with the file - this is only done when the reference
*     count drops to zero.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Must be a top-level object.
*     - Does not attempt execute if status is bad on entry.
*     - See the documentation for H5Fclose for details on what
*       happens if other locators are associated with the file.
*     - calls unlink(2) to remove the file from the file system.

*  History:
*     2014-09-18 (TIMJ):
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

#include <unistd.h>
#include <errno.h>

#include "hdf5.h"

#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

int
hdsErase(HDSLoc **locator, int *status) {
  int nlev = 0;
  char file_str[4096];   /* Consider using lower-level dynamic API */
  char path_str[1024];
  int errstat = 0;

  if (*status != SAI__OK) return *status;

  if ( !(*locator)->isprimary ) {
    *status = DAT__LOCIN;
    emsRep("hdsErase_1", "Must supply a top level primary locator to hdsErase",
           status );
    return *status;
  }

  /* unlink(2) requires that we know the path to the file.
   hdsTrace is fine for this but requires us to allocate the buffers
   in advance when we would really like to use a lower level API that
   knows the required length. */
  hdsTrace(*locator, &nlev, path_str, file_str, status,
           sizeof(path_str), sizeof(file_str) );

  /* Free the resources (will close the file) */
  datAnnul(locator, status);

  /* Now attempt to unlink the file if things are looking ok */
  if (*status != SAI__OK) return *status;

  errstat = unlink(file_str);
  if (*status == SAI__OK && errstat > 0) {
    *status = DAT__FILND;
    emsErrno( "ERRNO", errno );
    emsRepf("hdsErase_2", "Error unlinking file %s: ^ERRNO",
           status, file_str);
  }

  return *status;
}
