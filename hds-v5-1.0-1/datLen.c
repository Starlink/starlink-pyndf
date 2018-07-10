/*
*+
*  Name:
*     datLen

*  Purpose:
*     Enquire primitive length

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datLen( const HDSLoc * locator, size_t *len, int * status );

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Primitive locator.
*     len = size_t * (Returned)
*        Number of bytes per element
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*    Enquire the length of a primitive. In the case of a character object,
*    this is the number of characters per element. For other primitive types
*    it is the number of bytes per element.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-08-29 (TIMJ):
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
datLen( const HDSLoc * locator, size_t * clen, int * status ) {

  hid_t memtype = 0;
  hid_t h5type = 0;

  *clen = 1; /* force to one as default */

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datLen", 1, locator, 1, status );

  if (locator->dataset_id <= 0) {
    *status = DAT__OBJIN;
    emsRep("datClen_1",
           "Object is not primitive; the character string length is not defined "
           "(possible programming error)", status );
    return *status;
  }

  CALLHDF( h5type,
           H5Dget_type( locator->dataset_id ),
           DAT__HDF5E,
           emsRep("dat1Type_1", "datType: Error obtaining data type of dataset", status)
           );

  /* Number of bytes representing the type */
  memtype = dau1Native2MemType( h5type, status );
  *clen = H5Tget_size( memtype );

 CLEANUP:
  if (h5type) H5Tclose(h5type);
  if (memtype) H5Tclose(memtype);
  return *status;

}
