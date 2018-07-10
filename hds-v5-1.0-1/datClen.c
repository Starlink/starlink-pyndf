/*
*+
*  Name:
*     datClen

*  Purpose:
*     Obtain character string length

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datClen( const HDSLoc *locator, size_t *clen, int *status);

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Primitive object locator.
*     clen = size_t * (Returned)
*        Character string length.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     The routine returns the number of characters required to
*     represent the values of a primitive object. If the object is
*     character-type, then its length is returned directly. Otherwise,
*     the value returned is the number of characters required to format
*     the object's values (as a decimal string if appropriate) without
*     loss of information.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     -  The value returned by this routine is equal to the default
*     number of characters allocated to each element whenever a
*     primitive object is mapped using an access type of '_CHAR' (i.e.
*     without specifying the length to be used explicitly).
*     -  If this routine is called with STATUS set, then a value of 1
*     will be returned for the CLEN argument, although no further
*     processing will occur. The same value will also be returned if
*     the routine should fail for any reason.

*  History:
*     2014-08-27 (TIMJ):
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

/* For now rely on PRIMDAT */
#include "prm_par.h"

#include "dat_err.h"

int
datClen( const HDSLoc * locator, size_t * clen, int * status ) {

  hdstype_t htype;
  hid_t h5type = 0;

  *clen = 1; /* force to one as default */

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datClen", 1, locator, 1, status );

  if (locator->dataset_id <= 0) {
    *status = DAT__OBJIN;
    emsRep("datClen_1",
           "Object is not primitive; the character string length is not defined "
           "(possible programming error)", status );
    return *status;
  }

  /* Need to get the data type */
  htype = dat1Type( locator, status );
  if (*status != SAI__OK) goto CLEANUP;

  switch (htype) {
  case HDSTYPE_INTEGER:
    *clen = VAL__SZI;
    break;
  case HDSTYPE_REAL:
    *clen = VAL__SZR;
    break;
  case HDSTYPE_DOUBLE:
    *clen = VAL__SZD;
    break;
  case HDSTYPE_BYTE:
    *clen = VAL__SZB;
    break;
  case HDSTYPE_UBYTE:
    *clen = VAL__SZUB;
    break;
  case HDSTYPE_WORD:
    *clen = VAL__SZW;
    break;
  case HDSTYPE_UWORD:
    *clen = VAL__SZUW;
    break;
  case HDSTYPE_LOGICAL:
    *clen = 5; /* Primdat does not know but support "FALSE" */
    break;
  case HDSTYPE_INT64:
    *clen = VAL__SZK;
    break;
  case HDSTYPE_CHAR:
    /* length of the special type */
    CALLHDF( h5type,
             H5Dget_type( locator->dataset_id ),
             DAT__HDF5E,
             emsRep("datClen_3", "datClen: Error obtaining data type of dataset", status)
             );
    *clen = H5Tget_size( h5type );
    break;

  default:
    *clen = 1;
    *status =  DAT__TYPIN;
    emsRep("datClen_2", "datClen: Unexpected type when obtaining character length",
           status );

  }

 CLEANUP:
  if (h5type > 0) H5Tclose(h5type);

  return *status;
}
