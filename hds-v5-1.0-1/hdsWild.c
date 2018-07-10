/*
*+
*  Name:
*     hdsWild

*  Purpose:
*     Perform a wild-card search for HDS container files

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hdsWild(const char *fspec, const char *mode, HDSWild **iwld,
*             HDSLoc **loc, int *status);

*  Arguments:
*     fspec = const char * (Given)
*        The wild-card specification identifying the container files required
*        (a default file type extension of `.sdf' is assumed, if not specified).
*        The syntax of this specification depends on the host operating system.
*     mode = const char * (Given)
*        The mode of access required to the container files: 'READ', 'UPDATE'
*        or 'WRITE' (case insensitive).
*     iwld = HDSWild ** (Returned)
*        Context. This part of the API is subject to change.
*     loc = HDSLoc ** (Returned)
*        A primary locator to the top-level object in the next container file
*        to satisfy the file specification given in FSPEC. A value of NULL
*        will be returned (without error) if no further container files remain
*        to be located.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     The routine searches for HDS container files whose names match a
*     given wild-card file specification, and which are accessible using a
*     specified mode of access. It is normally called repeatedly, returning
*     a locator for the top-level object in a new container file on each
*     occasion, and a null locator value when no more container
*     files remain to be located.

*     A call to hdsEwild should be made to annul the search context
*     identifier when the search is complete. This will release any
*     resources used.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Not Yet Implemented.
*     - C API is unstable and has not been properly designed.

*  History:
*     2014-10-17 (TIMJ):
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

#include "dat_err.h"

int
hdsWild(const char *fspec, const char *mode, HDSWild **iwld,
        HDSLoc **loc, int *status) {

  if (*status != SAI__OK) return *status;

  *status = DAT__FATAL;
  emsRep("hdsWild", "hdsWild: Not yet implemented for HDF5",
         status);

  return *status;


}
