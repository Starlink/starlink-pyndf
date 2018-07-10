/*
*+
*  Name:
*     datBasic

*  Purpose:
*     Map primitive as basic units

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datBasic(const HDSLoc *locator, const char *mode_c, unsigned char **pntr,
*              size_t *len, int *status);

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Primitive locator.
*     mode_c = const char * (Given)
*        Access mode ("READ", "UPDATE", or "WRITE").
*     pntr = void ** (Returned)
*        Pointer to the mapped value.
*     len = size_t * (Returned)
*        Total number of bytes mapped.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Maps a primitive as a sequence of basic machine units (bytes) for reading,
*     writing or updating.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Does not attempt to interpret the bytes.
*     - Not implemented in HDF5 interface.

*  History:
*     2014-10-14 (TIMJ):
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

#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

int
datBasic(const HDSLoc *locator, const char *mode_c, unsigned char **pntr,
         size_t *len, int *status) {

  *len = 0;
  *pntr = NULL;

  if (*status == SAI__OK) return *status;

  *status = SAI__ERROR;
  emsRep("datBasic", "datBasic is not available in HDF5 interface",
         status);

  return *status;

}

