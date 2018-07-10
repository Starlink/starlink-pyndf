/*
*+
*  Name:
*    dat1ImportFloc

*  Purpose:
*    Import a fortran HDS locator buffer into C

*  Invocation:
*    clocator = dat1ImportFloc( const char flocator[DAT__SZLOC], int len, int * status);

*  Description:
*    This function should be used to convert a Fortran HDS locator
*    (implemented as a string buffer) to a C locator struct.

*  Arguments
*    flocator = const char * (Given)
*       Fortran character string buffer. Should be at least DAT__SZLOC
*       characters long.
*    len = int (Given)
*       Size of Fortran character buffer. Sanity check.
*    status = int * (Given and Returned)
*       Inherited status. Attempts to execute even if status is not SAI__OK
*       on entry.

*  Returned Value:
*    clocator = HDSLoc *
*       C HDS locator corresponding to the Fortran locator.
*       Should be freed by using datAnnul().

*  Authors:
*    Tim Jenness (JAC, Hawaii)
*    David Berry (JAC, Preston)

*  History:
*     2014-09-07 (TIMJ):
*        Initial version

*  Notes:
*    - Does not check the contents of the locator for validity but does check for
*    common Fortran error locators such as DAT__ROOT and DAT__NOLOC.
*    - For internal usage by HDS only. The public version is datImportFloc.
*    - Differs from the original HDS API in that it returns the locator.
*    - Attempts to execute even if status is bad.
*    - To allow this library to be a drop in replacement for HDS without requiring
*      a recompile, DAT__SZLOC must match the standard HDS value.

*  See Also:
*    - datImportFloc
*    - datExportFloc

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

#include "ems.h"

#include "hds1.h"
#include "dat1.h"
#include "dat_err.h"
#include "sae_par.h"

HDSLoc *
dat1ImportFloc ( const char flocator[DAT__SZLOC], int loc_length, int * status) {

  long ptr_as_long = 0;
  HDSLoc * clocator = NULL;

  /* Validate the locator length. */
  if (loc_length != DAT__SZLOC ) {
    if (*status == SAI__OK ) {
       *status = DAT__LOCIN;
       emsRepf( "DAT1_IMPORT_FLOC", "Locator length is %d not %d", status,
                loc_length, DAT__SZLOC);
    }
    return NULL;
  };

  /* Check obvious error conditions */
  if (strncmp( DAT__ROOT, flocator, loc_length) == 0 ){
    if( *status == SAI__OK ) {
       *status = DAT__LOCIN;
       emsRep( "dat1ImportFloc_ROOT",
               "Input HDS Locator corresponds to DAT__ROOT but that can only be used from NDF",
               status );
    }
    return NULL;
  }

  /* Check obvious error conditions */
  if (strncmp( DAT__NOLOC, flocator, loc_length) == 0 ){
    if( *status == SAI__OK ) {
       *status = DAT__LOCIN;
       emsRep( "datImportFloc_NOLOC",
               "Input HDS Locator corresponds to DAT__NOLOC but status is good (Possible programming error)",
               status );
    }
    return NULL;
  }

  /* Everything seems to be okay so now convert the string buffer to the
     required pointer. We ignore status as sometimes we need to try
     to get the value regardless (otherwise DAT_ANNUL from Fortran would
     never succeed). */

  ptr_as_long = strtol( flocator, NULL, 16 );

  if (ptr_as_long == 0) {
    /* This should not have happened */
    if (*status == SAI__OK) {
      *status = DAT__LOCIN;
      emsRep("dat1_import_floc_3",
             "Error importing locator from Fortran", status );
      return NULL;
    }
  }

  /* Do the cast */
  clocator = (HDSLoc *)ptr_as_long;
  return clocator;
}

