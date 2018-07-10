/*
*+
*  Name:
*     datName

*  Purpose:
*     Enquire the object name

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datName( const HDSLoc * locator, char name_str[DAT__SZNAM+1],
*              int * status );

*  Arguments:
*     locator = const HDSLoc * (Given)
*         Object locator
*     name_str = char * (Given and Returned)
*         Buffer to receive the object name. Must be of size
*         DAT__SZNAM+1.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Enquire the object name.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-08-26 (TIMJ):
*        Initial version
*     2014-11-22 (TIMJ):
*        Support use of HDF5 root group as HDS root
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
#include <string.h>

#include "hdf5.h"

#include "star/one.h"
#include "ems.h"
#include "star/util.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"
#include "sae_par.h"

int
datName(const HDSLoc *locator,
        char name_str[DAT__SZNAM+1],
        int *status) {

  hid_t objid;
  ssize_t lenstr;
  char * tempstr = NULL;
  char * cleanstr = NULL;

  /* Store something in there as a placeholder in case
     something goes wrong */
  star_strlcpy( name_str, "<<ERROR>>", DAT__SZNAM+1 );

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datName", 1, locator, 1, status );

  objid = dat1RetrieveIdentifier( locator, status );
  if (*status != SAI__OK) return *status;

  /* Get the full name */
  tempstr = dat1GetFullName( objid, 0, &lenstr, status );

  /* Handle the presence of a HDF5 array structure path
     that needs to be converted to HDS hierarchy */
  cleanstr = dat1FixNameCell( tempstr, status );

  if (cleanstr) {
    /* update the string length */
    lenstr = strlen(cleanstr);
  } else {
    /* just copy the pointer, will not free it later */
    cleanstr = tempstr;
  }

  /* Now walk through the string backwards until we find the
     "/" character indicating the parent group */
  if (*status == SAI__OK) {
    ssize_t i;
    ssize_t startpos = 0; /* whole string as default */
    for (i = 0; i <= lenstr; i++) {
      size_t iposn = lenstr - i;
      if ( cleanstr[iposn] == '/' ) {
        startpos = iposn + 1; /* want the next character */
        break;
      }
    }
    /* Now copy what we need unless this is the  root group */
    if (lenstr == 1) {
      /* Must read the attribute */
      dat1NeedsRootName( locator->group_id, HDS_FALSE, name_str, DAT__SZNAM+1, status );
    } else {
      one_strlcpy( name_str, &(cleanstr[startpos]), DAT__SZNAM+1, status );
    }
  }

  if (tempstr != cleanstr) MEM_FREE(cleanstr);
  if (tempstr) MEM_FREE(tempstr);

  if (*status != SAI__OK) {
    emsRep("datName_4", "datName: Error obtaining a name of a locator",
           status );
  }

  return *status;
}
