/*
*+
*  Name:
*     dau1HdsType

*  Purpose:
*     Enquire the HDS type of an HDF5 datatype (enum version)

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hdstype_t dau1HdsType( hid_t h5type, int * status );

*  Arguments:
*     h5type = hid_t (Given)
*        HDF5 data type.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Returned Value:
*     type = hdstype_t
*        Enum indicating the HDS type associated with this
*        HDF5 type. Returns HDSTYPE_NONE on error.

*  Description:
*     Enquire the HDS type of an HDF5 object and returns that type as an enum.
*     use datType() to return a string form that will also return types
*     of structures and details of string length.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - use datType to get the string form from a locator.
*     - use dat1Type to get this enum form from a locator.

*  History:
*     2014-09-15 (TIMJ):
*        Initial version, substantially copied from dat1Type.
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

#include "star/one.h"
#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"
#include "dat_err.h"

hdstype_t
dau1HdsType( hid_t h5type, int * status ) {

  H5T_class_t tclass = 0;
  size_t dsize = 0;
  hdstype_t thetype = HDSTYPE_NONE;

  if (*status != SAI__OK) return thetype;

  CALLHDF( tclass,
           H5Tget_class( h5type ),
           DAT__HDF5E,
           emsRep("dat1Type_2", "datType: Error obtaining class of data type", status)
           );

  /* Number of bytes representing the type */
  dsize = H5Tget_size( h5type );

  if (*status == SAI__OK) {
    switch (tclass) {
    case H5T_INTEGER:
      {
        /* then need to know signed or unsigned int */
        H5T_sign_t dsign = H5Tget_sign( h5type );
        if (dsign < 0) {
          *status = DAT__HDF5E;
          emsRep("dat1Type_3", "datType: Error obtaining sign of an integer type", status );
        goto CLEANUP;
        }
        if (dsign == H5T_SGN_NONE) {
          if ( dsize == 1 ) {
            thetype = HDSTYPE_UBYTE;
          } else if (dsize == 2) {
            thetype = HDSTYPE_UWORD;
          } else {
            *status = DAT__TYPIN;
            emsRepf("dat1Type_3a",
                    "Unexpected number of bytes (%zu) in unsigned integer type",
                    status, dsize);
          }
        } else {
          /* Signed types */
          switch (dsize) {
          case 1:
            thetype = HDSTYPE_BYTE;
            break;
          case 2:
            thetype = HDSTYPE_WORD;
            break;
          case 4:
            thetype = HDSTYPE_INTEGER;
            break;
          case 8:
            thetype = HDSTYPE_INT64;
            break;
          default:
            *status = DAT__TYPIN;
            emsRepf("dat1Type_3b", "datType: Unexpected number of bytes in integer (%zu)",
                    status, dsize);
          }
        }
      }
      break;
    case H5T_FLOAT:
      if ( dsize == 4 ) {
        thetype = HDSTYPE_REAL;
      } else if (dsize == 8) {
        thetype = HDSTYPE_DOUBLE;
      } else {
        *status = DAT__FATAL;
        emsRepf("datType_5", "Error reading size of float data type. Got %zu bytes"
                " but only understand 4 and 8", status, dsize);
      }
      break;

    case H5T_STRING:
      thetype = HDSTYPE_CHAR;
      break;

    case H5T_BITFIELD:
      if ( dsize == 1 || dsize == 4 ) { /* on disk and in memory version */
        thetype = HDSTYPE_LOGICAL;
      } else {
        *status = DAT__FATAL;
        emsRepf("datType_5", "Error reading size of logical data type. Got %zu bytes"
                " but only understand 1 or 4", status, dsize);
      }
      break;

    default:
      *status = DAT__TYPIN;
      emsRep("datType_4", "dat1Type: Unexpected type class from dataset", status);
    }
  }

 CLEANUP:
  return thetype;

}
