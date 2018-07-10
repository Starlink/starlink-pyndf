/*
*+
*  Name:
*     dau1Native2MemType

*  Purpose:
*     Map on-disk type to in-memory type

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     memtype = dau1Native2MemType( hid_t nativetype, int * status );

*  Arguments:
*     nativetype = hid_t (Given)
*        Type representing the on disk format for a primitive type.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Returned Value:
*    memtype = hid_t
*        In-memory data type. Should be freed with H5Tclose.

*  Description:
*    For some HDS data types the on-disk data type is not the same
*    as the in-memory data type. This routine is given the on-disk
*    data type and will return the corresponding data type.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - _LOGICAL data types are stored on disk in 8-bit bitfields
*       but in memory are 32-bit integers. HDS inherits this from
*       Fortran where a LOGICAL type is 32-bits.

*  History:
*     2014-09-22 (TIMJ):
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

hid_t
dau1Native2MemType( hid_t nativetype, int * status ) {
  hdstype_t htype = HDSTYPE_NONE;
  hid_t rettype = 0;
  if (*status != SAI__OK) return 0;

  htype = dau1HdsType( nativetype, status );
  if (*status != SAI__OK) return 0;

  if ( htype == HDSTYPE_LOGICAL ) {
    /* _LOGICAL is a 32bit number but we store in HDF5 in 8 bits */
    size_t szbool = sizeof(hdsbool_t);
    switch (szbool) {
    case 4:
      rettype = H5T_NATIVE_B32;
      break;
    case 2:
      rettype = H5T_NATIVE_B16;
      break;
    case 1:
      rettype = H5T_NATIVE_B8;
      break;
    default:
      *status = DAT__TYPIN;
      emsRep("dau1Native2MemType", "Unexpected size of _LOGICAL type"
             " (possible programming error)", status );
      return 0;
    }
  } else {
    rettype = nativetype;
  }

  /* But we promise to copy the type so that it is clear that the
     caller should free it */
  return H5Tcopy(rettype);
}
