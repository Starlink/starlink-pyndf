/*
*+
*  Name:
*     datExportFloc

*  Purpose:
*     Export from a C HDS Locator to a Fortran HDS locator

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datExportFloc( HDSLoc** clocator, int free, int len, char flocator[DAT__SZLOC],
*                    int * status );

*  Arguments:
*     clocator = HDSLoc ** (Given and Returned)
*        Pointer to the locator to be exported. See the "free" description below
*        as to whether this locator will be annulled or not.
*     free = int (Given)
*        If true (1) the C locator is annuled one the Fortran locator is populated.
*        If false, the locator memory is not touched.
*     len = int (Given)
*        Size of the fortran character buffer to receive the locator. Sanity check
*        and should be DAT__SZLOC.
*     flocator = char [DAT__SZLOC] (Returned)
*       Fortran character string buffer. Should be at least DAT__SZLOC
*       characters long. If clocator is NULL, fills the buffer
*       with DAT__NOLOC.
*     status = int* (Given and Returned)
*        Pointer to global status.  If status is bad the Fortran locator will be
*        filled with DAT__NOLOC. The memory associated with clocator will
*        be freed if free is true regardless of status.

*  Description:
*     This function should be used to populate a Fortran HDS locator buffer
*     (usually a Fortran CHARACTER string of size DAT__SZLOC) from a C HDS
*     locator structure. This function is also available in the
*     HDS_EXPORT_CLOCATOR macro defined in hds_fortran.h.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*    - Fortran locator string must be preallocted. The C locator can be freed
*    at the discretion of the caller. This simplifies code in the Fortran
*    interface wrapper.
*    - This routine is intended to be used solely for
*    wrapping Fortran layers from C. "Export" means to export a native
*    C locator to Fortran.
*    - There is no Fortran eqiuvalent to this routine.
*    - This routine differs from the HDS equivalent in that the address
*    of the supplied pointer is stored in the Fortran string buffer and not the contents
*    of the struct. This is done to constrain the required size of the locator
*    in Fortran to allow this library to be used as a drop in replacement for
*    HDS without requiring a recompile.

*  History:
*     2014-09-07 (TIMJ):
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

#include <string.h>

#include "hdf5.h"

#include "ems.h"
#include "sae_par.h"
#include "star/one.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"
#include "hds_fortran.h"

void datExportFloc ( HDSLoc **clocator, int free, int loc_length, char flocator[DAT__SZLOC], int * status) {

  /* Validate the locator length */
  if (*status == SAI__OK && loc_length != DAT__SZLOC ) {
    *status = DAT__LOCIN;
    emsRepf( "datExportFloc", "Locator length is %d not %d", status,
            loc_length, DAT__SZLOC);
  }

  /* if everything is okay we store the pointer location in the Fortran
     locator */
  if ( *status == SAI__OK && *clocator != NULL ) {

    /* We export from C by storing the pointer of the C struct in the
       Fortran character buffer. We can not store a clone in the Fortran
       locator because clones are documented to not clone mapped status
       and DAT_MAP / DAT_UNMAP will fail for clones. We just store the
       supplied locator and null out the C version if we are being requested
       to free it. Note that if we free=false the caller should not then
       annul the locator as that would mess up the Fortran side. If the current
       scheme does not work, we could try assigning the clone to the caller and
       the original to the fortran locator but this requires some thought. */

    one_snprintf(flocator, loc_length, "%p", status, *clocator );

  } else {
    strncpy( flocator, DAT__NOLOC, DAT__SZLOC );
  }

  /* Null out the caller if requested. Do not annul as we have stored
     the original pointer in the Fortran layer */
  if (free) *clocator = NULL;

  return;
}
