/*
*+
*  Name:
*     datNew

*  Purpose:
*     Create a new component in a structure

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     int datNew( const HDSLoc *locator, const char *name_str, const char *type_str,
*                 int ndim, const hdsdim dims[], int * status );

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Locator to structure that will receive the new component.
*     name = const char * (Given)
*        Name of the object in the container.
*     type = const char * (Given)
*        Type of object.  If type matches one of the HDS primitive type names a primitive
*        of that type is created, otherwise the object is assumed to be a structure.
*     ndim = int (Given)
*        Number of dimensions. Use 0 for a scalar. See the Notes for a discussion of
*        arrays of structures.
*     dims = const hdsdim [] (Given)
*        Dimensionality of the object. Should be dimensioned with ndim. The array
*        is not accessed if ndim == 0.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Creates a new component (primitive type or structure) in an existing structure.

*  Returned Value:
*     int = inherited status on exit. This is for compatibility with the original
*           HDS API.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Use datFind to obtain a locator to this component.

*  History:
*     2014-08-20 (TIMJ):
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
#include "hds.h"
#include "sae_par.h"

int
datNew( const HDSLoc    *locator,
        const char      *name_str,
        const char      *type_str,
        int       ndim,
        const hdsdim    dims[],
        int       *status) {

  HDSLoc * newloc;

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datNew", 1, locator, 0, status );

  newloc = dat1New( locator, 0, name_str, type_str, ndim, dims, status );

  /* Free the locator as datNew does not expect you to use the
     component you have just created */
  datAnnul( &newloc, status );

  return *status;

}
