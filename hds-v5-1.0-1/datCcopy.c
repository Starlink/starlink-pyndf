/*
*+
*  Name:
*     datCcopy

*  Purpose:
*     Copy one structure level

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datCcopy(const HDSLoc *locator1, const HDSLoc *locator2, const char *name,
         HDSLoc **locator3, int *status );

*  Arguments:
*     locator1 = const HDSLoc * (Given)
*        Object locator to copy.
*     locator2 = const HDSLoc * (Given)
*        Locator of structure to receive copy of object.
*     name = const char * (Given)
*        Name of object when copied into structure.
*     locator3 = HDSLoc ** (Returned)
*        Locator of newly copied component.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Copy an object into a structure and give the new component the
*     specified name. If the source object is a structure, a new structure
*     of the same type and shape is created but the content of the
*     original structure is not copied.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-10-14 (TIMJ):
*        Initial version
*     2014-10-29 (TIMJ):
*        Enable copying of an undefined primitive object.
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
datCcopy(const HDSLoc *locator1, const HDSLoc *locator2, const char *name,
         HDSLoc **locator3, int *status ) {
  char type_str[DAT__SZTYP+1];
  hdsdim hdims[DAT__MXDIM];
  int ndims;

  if (*status != SAI__OK) return *status;

  /* Validate input locators. */
  dat1ValidateLocator( "datCcopy", 1, locator1, 1, status );
  dat1ValidateLocator( "datCcopy", 1, locator2, 0, status );

  if (dat1IsStructure(locator1, status)) {

    /* need the type and dimensionality of the structure to create
       in new location */
    datType( locator1, type_str, status );
    datShape( locator1, DAT__MXDIM, hdims, &ndims, status );

    *locator3 = dat1New( locator2, 0, name, type_str, ndims, hdims, status );

  } else {
    hdsbool_t state = 0;
    /* We only copy if the primitive object is defined */
    datState( locator1, &state, status );
    if ( state ) {
      datCopy( locator1, locator2, name, status );
    } else {
      /* Undefined so just make something of the right shape and type */
      datType( locator1, type_str, status );
      datShape( locator1, DAT__MXDIM, hdims, &ndims, status );
      datNew( locator2, name, type_str, ndims, hdims, status );
    }

    /* and get a locator to the copied entity */
    datFind( locator2, name, locator3, status );

  }

  return *status;
}
