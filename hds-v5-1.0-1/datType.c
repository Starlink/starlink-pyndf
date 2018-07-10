/*
*+
*  Name:
*     datType

*  Purpose:
*     Enquire the type of an object

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datType( const HDSLoc *locator, char type_str[DAT__SZTYP+1], int * status );

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Object locator
*     type_str = char * (Returned)
*        Buffer of size DAT__SZTYP+1 to receive the object type.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Enquire the tyope of an object

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

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

#include <string.h>

#include "hdf5.h"

#include "star/one.h"
#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"
#include "dat_err.h"

int
datType( const HDSLoc *locator, char type_str[DAT__SZTYP+1], int * status ) {

  hdstype_t hdstyp = HDSTYPE_NONE;
  hid_t h5type = 0;
  hid_t h5attr = 0;

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datType", 1, locator, 1, status );

  hdstyp = dat1Type( locator, status );
  if (*status != SAI__OK) return *status;

  switch (hdstyp) {
  case HDSTYPE_INTEGER:
    one_strlcpy( type_str, "_INTEGER", DAT__SZTYP+1, status);
    break;
  case HDSTYPE_REAL:
    one_strlcpy( type_str, "_REAL", DAT__SZTYP+1, status);
    break;
  case HDSTYPE_DOUBLE:
    one_strlcpy( type_str, "_DOUBLE", DAT__SZTYP+1, status);
    break;
  case HDSTYPE_BYTE:
    one_strlcpy( type_str, "_BYTE", DAT__SZTYP+1, status);
    break;
  case HDSTYPE_UBYTE:
    one_strlcpy( type_str, "_UBYTE", DAT__SZTYP+1, status);
    break;
  case HDSTYPE_WORD:
    one_strlcpy( type_str, "_WORD", DAT__SZTYP+1, status);
    break;
  case HDSTYPE_UWORD:
    one_strlcpy( type_str, "_UWORD", DAT__SZTYP+1, status);
    break;
  case HDSTYPE_LOGICAL:
    one_strlcpy( type_str, "_LOGICAL", DAT__SZTYP+1, status);
    break;
  case HDSTYPE_INT64:
    one_strlcpy( type_str, "_INT64", DAT__SZTYP+1, status);
    break;
  case HDSTYPE_CHAR:
    /* Get the type from the dataset and request its size */
    {
      size_t dsize = 0;
      CALLHDF( h5type,
               H5Dget_type( locator->dataset_id ),
               DAT__HDF5E,
               emsRep("datType_1", "datType: Error obtaining data type of dataset", status)
               );
      dsize = H5Tget_size( h5type );
      one_snprintf( type_str, DAT__SZTYP+1, "_CHAR*%zu", status, dsize );
    }
    break;
  case HDSTYPE_STRUCTURE:
    /* Read the type attribute */
    dat1GetAttrString( locator->group_id, HDS__ATTR_STRUCT_TYPE, HDS_TRUE,
                       "HDF5NATIVEGROUP", type_str, DAT__SZTYP+1, status );
    break;
  default:
    *status = DAT__TYPIN;
    emsRepf("datType_inv","datType: Unknown type associated with dataset/group (%d)",
            status, hdstyp);
  }

 CLEANUP:
  if (h5type > 0) H5Tclose(h5type);
  if (h5attr > 0) H5Aclose(h5attr);
  return *status;

}
