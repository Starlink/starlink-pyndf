/*
*+
*  Name:
*     hdsCopy

*  Purpose:
*     Copy an object to a new container file.

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hdsCopy( const HDSLoc *locator, const char *file_str,
*              const char name_str[DAT__SZNAM], int *status );
*  Arguments:
*     locator = const HDSLoc * (Given)
*        Locator for object to be copied.
*     file_str = const char * (Given)
*        Name of the new container file to be created.
*     name_str = const char [DAT__SZNAM] (Given)
*        Name which the new top-level object is to have.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     The routine makes a copy of an HDS object, placing the copy in a
*     new container file (which is created), as the top-level
*     object. The copying operation is recursive; i.e. all
*     sub-components of a structure will also be copied.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Not Yet Implemented.
*     - The routine makes a copy of an HDS object, placing the copy in a
        new container file (which is created), as the top-level
        object. The copying operation is recursive; i.e. all
        sub-components of a structure will also be copied.

*  History:
*     2014-10-16 (TIMJ):
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
hdsCopy( const HDSLoc *locator, const char *file_str,
         const char name_str[DAT__SZNAM], int *status ) {

  if (*status != SAI__OK) return *status;

  *status = DAT__FATAL;
  emsRep("hdsCopy", "hdsCopy: Not yet implemented for HDF5",
         status);

  return *status;
}
