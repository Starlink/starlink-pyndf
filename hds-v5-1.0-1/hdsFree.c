/*
*+
*  Name:
*     hdsFree

*  Purpose:
*     Free container file

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hdsFree(const HDSLoc *locator, int *status);

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Locator to the container file's top-level object.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Release all explicit or implicit locks on a container file,
*     thereby granting write-access to other processes.
*
*     This function is not yet implemented. See datLock for an
*     alternative locking mechanism based on POSIX threads.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Flushes buffers to disk. This is consistent with HDSv4 but
*       is not obviously mentioned in the documentation.
*     - Should be called to free resources locked via hdsLock.

*  History:
*     2014-10-17 (TIMJ):
*        Initial version
*     2014-11-18 (TIMJ):
*        Add call to flush buffers to disk.
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
hdsFree(const HDSLoc *locator, int *status) {

  if (*status != SAI__OK) return *status;

  /* It seems that HDSv4 flushes buffers as well as unlocking
     the file. This despite no hdsLock ever having been called */
  H5Fflush( locator->file_id, H5F_SCOPE_LOCAL );

  /* Documented to do nothing if hdsLock has not been called
     so it is safe to do nothing here as we know that hdsLock
     can not have been called since it is not implemented yet. */
  return *status;
}
