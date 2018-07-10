/*
*+
*  Name:
*     datThere

*  Purpose:
*     Enquire if a component of a structure exists.

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datThere( const HDSLoc * locator, const char * name, hdsbool_t *there,
*               int *status );

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Structure locator
*     name = const char * (Given)
*        Component name to check
*     there = hdsbool_t * (Returned)
*        Boolean indicating existence: 1 if exists, 0 otherwise.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Enquire if a component of a structure exists.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-08-28 (TIMJ):
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
datThere( const HDSLoc * locator, const char * name, hdsbool_t *there,
          int *status ) {

  htri_t exists = 0;
  char cleanname[DAT__SZNAM+1];

  *there = 0;
  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datThere", 1, locator, 1, status );

  /* containing locator must refer to a group */
  if (locator->group_id <= 0) {
    *status = DAT__OBJIN;
    emsRep("datFind_1", "datThere: Input object is not a structure",
           status);
    return *status;
  }

  /* Normalize the name string */
  dau1CheckName( name, 1, cleanname, sizeof(cleanname), status );
  if (*status != SAI__OK) return *status;

  exists = H5Lexists( locator->group_id, cleanname, H5P_DEFAULT);

  if (exists < 0) {
    *status = DAT__HDF5E;
    emsRepf("", "datThere: Error checking existence of component '%s'",
            status, cleanname);
  } else {
    /* HDF5 API indicates TRUE and FALSE as being actual things but
       the public API does not actually include them so we assume
       positive means true. */
    *there = (exists > 0 ? 1 : 0 );
  }

  return *status;
}
