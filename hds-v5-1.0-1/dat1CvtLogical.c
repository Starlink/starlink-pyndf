/*
*+
*  Name:
*     dat1CvtLogical

*  Purpose:
*     Translate data to or from logical format

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     dat1CvtChar( size_t nval, hdstype_t intype, size_t nbin,
*                  hdstype_t outtype, size_t nbout, const void * imp, void * exp,
*                  size_t *nbad, int * status );

*  Arguments:
*     nval = size_t (Given)
*        Number of values to be converted.
*     intype = hdstype_t (Given)
*        Type of data in "imp" array.
*     nbin = size_t (Given)
*        Number of bytes per input element. For strings this will be _CHAR*nbin.
*     outtype = hdstype_t (Given)
*        Required type of output data array "exp".
*     nbout = size_t (Given)
*        Number of bytes per output element. For strings this will be _CHAR*nbout.
*     imp = void * (Given)
*        Buffer with data to be converted. nval elements of type
*        intype.
*     exp = void * (Returned)
*        Buffer to receive converted data. nval elements of type
*        outtype.
*     nbad = size_t * (Returned)
*        Number of bad data conversions encountered.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     This routine 'translates' a contiguous sequence of data values from one
*     location to another. It is only intended for conversions to and from
*     logical formats and is relatively inefficient.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - DAT__CONER error status is preserved for conversion that result
*       in the insertion of a "bad" value. This allows errors to be handled
*       by the caller. Other errors will use different codes.
*     - During conversion, any data values that cannot be sensibly
*       translated from the source type to the destination type are substituted
*       by a specific 'bad' value, and the return status set accordingly.
*     - All numeric input types are converted to output logicals by using the
*       HDS definition of true being equivalent to bit 0 being set.
*     - All numeric output types are simply 1 or 0 depending on the output of
*       the HDS_ISTRUE macro.
*     - Logical conversions to or from character strings are handled by dat1CvtChar.
*     - HDS does not treat bad values as special during conversion so a bad logical
*       value will be ignored and input bad numeric values are treated as true or false
*       dependent on bit 0. This might be a bug.

*  History:
*     2014-09-15 (TIMJ):
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
#include <stdio.h>
#include <limits.h>
#include <float.h>

#include "hdf5.h"

#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

int
dat1CvtLogical( size_t nval, hdstype_t intype, size_t nbin,
             hdstype_t outtype, size_t nbout, const void * imp, void * exp,
             size_t *nbad, int * status ) {
  size_t n;
  char * buffer = NULL;
  HdsTypeInfo *typeinfo;

  *nbad = 0;
  if (*status != SAI__OK) return *status;

  /* Sanity check */
  if (intype != HDSTYPE_LOGICAL && outtype != HDSTYPE_LOGICAL) {
    *status = DAT__TYPIN;
    emsRep("dat1CvtLogical", "dat1CvtLogical can not do arbitrary type conversion."
           " (Possible programming error)", status );
    return *status;
  }

  /* if the types are the same and the sizes are the same,
     just copy directly. ie both strings of type _LOGICAL */
  if (intype == outtype && nbin == nbout) {
    memmove( exp, imp, nval * nbin );
    return *status;
  }

  if (intype == HDSTYPE_LOGICAL && outtype == HDSTYPE_LOGICAL) {
    *status = DAT__TYPIN;
    emsRep("dat1CvtChar_2", "Should already have handled logical -> logical conversion"
           " (Possible programming error)", status);
    return *status;
  }

  if ( intype == HDSTYPE_CHAR || outtype == HDSTYPE_CHAR ) {
    /* Handled by dat1CvtChar so just punt */
    return dat1CvtChar( nval, intype, nbin, outtype, nbout, imp, exp, nbad, status );
  }


  /* Get cached type information */
  typeinfo = dat1TypeInfo();

  /* Convert each value, one at a time -- this is the case for string
     given, and non-string returned. Note that we convert EVERY element
     and set bad status if we end up having any bad elements inserted.
     We do not stop the conversion on the first bad copy. */
  if (intype == HDSTYPE_LOGICAL) {
    const hdsbool_t * inbuf = NULL;

    /* The input is a logical and we have to map that
       a numeric type. */
    inbuf = imp;

    for (n = 0; n < nval; n++) {

      switch( outtype ) {
      case HDSTYPE_INTEGER:
        ((int *)exp)[n] = ( HDS_ISTRUE( inbuf[n] ) ? 1 : 0 );
        break;
      case HDSTYPE_REAL:
        ((float *)exp)[n] = ( HDS_ISTRUE( inbuf[n] ) ? 1.0 : 0.0 );
        break;
      case HDSTYPE_DOUBLE:
        ((double *)exp)[n] = ( HDS_ISTRUE( inbuf[n] ) ? 1.0 : 0.0 );
        break;
      case HDSTYPE_INT64:
        ((int64_t *)exp)[n] = ( HDS_ISTRUE( inbuf[n] ) ? 1 : 0 );;
        break;
      case HDSTYPE_BYTE:
        ((char *)exp)[n] = ( HDS_ISTRUE( inbuf[n] ) ? 1 : 0 );
        break;
      case HDSTYPE_UBYTE:
        ((unsigned char *)exp)[n] = ( HDS_ISTRUE( inbuf[n] ) ? 1 : 0 );
        break;
      case HDSTYPE_WORD:
        ((short *)exp)[n] = ( HDS_ISTRUE( inbuf[n] ) ? 1 : 0 );
        break;
      case HDSTYPE_UWORD:
        ((unsigned short *)exp)[n] = ( HDS_ISTRUE( inbuf[n] ) ? 1 : 0 );
        break;
      case HDSTYPE_LOGICAL:
      case HDSTYPE_CHAR:
        /* handled previously and we should not be here */
        if (*status == SAI__OK) {
          *status = DAT__WEIRD;
          emsRep("dat1CvtLogical_internal",
                 "Internal consistency error on logical conversion", status );
          goto CLEANUP;
        }
        break;
      default:
        if (*status == SAI__OK) {
          *status = DAT__TYPIN;
          emsRepf("dat1CvtLogical_exp", "dat1CvtLogical: Unsupported output data type %d",
                  status, outtype);
          /* Never going to be resolved */
          goto CLEANUP;
        }
      }
    }
  } else if (outtype == HDSTYPE_LOGICAL) {
    hdsbool_t * outbuf = NULL;

    /* each value is converted one element at a time. */
    outbuf = exp;

    for (n = 0; n < nval; n++) {

      switch( intype ) {

      case HDSTYPE_INTEGER:
        {
          int inval = ((int *)imp)[n];
          outbuf[n] = ( inval & 1 ? HDS_TRUE : HDS_FALSE );
        }
        break;
      case HDSTYPE_REAL:
        {
          int inval = (int)((float *)imp)[n];
          outbuf[n] = ( inval & 1 ? HDS_TRUE : HDS_FALSE );
        }
        break;
      case HDSTYPE_DOUBLE:
        {
          int inval = (int)((double *)imp)[n];
          outbuf[n] = ( inval & 1 ? HDS_TRUE : HDS_FALSE );
        }
        break;
      case HDSTYPE_INT64:
        {
          int64_t inval = ((int64_t *)imp)[n];
          outbuf[n] = ( inval & 1 ? HDS_TRUE : HDS_FALSE );
        }
        break;
      case HDSTYPE_BYTE:
        {
          char inval = ((char *)imp)[n];
          outbuf[n] = ( inval & 1 ? HDS_TRUE : HDS_FALSE );
        }
        break;
      case HDSTYPE_UBYTE:
        {
          unsigned char inval = ((unsigned char *)imp)[n];
          outbuf[n] = ( inval & 1 ? HDS_TRUE : HDS_FALSE );
        }
        break;
      case HDSTYPE_WORD:
        {
          short inval = ((short *)imp)[n];
          outbuf[n] = ( inval & 1 ? HDS_TRUE : HDS_FALSE );
        }
        break;
      case HDSTYPE_UWORD:
        {
          unsigned short inval = ((unsigned short *)imp)[n];
          outbuf[n] = ( inval & 1 ? HDS_TRUE : HDS_FALSE );
        }
        break;
      case HDSTYPE_CHAR:
      case HDSTYPE_LOGICAL:
        /* handled previously and we should not be here */
        if (*status == SAI__OK) {
          *status = DAT__WEIRD;
          emsRep("dat1CvtLogical_internal2",
                 "Internal consistency error on logical conversion", status );
          goto CLEANUP;
        }
        break;
      default:
        if (*status == SAI__OK) {
          *status = DAT__TYPIN;
          emsRepf("dat1CvtChar_exp2", "dat1CvtChar: Unsupported output data type %d",
                  status, outtype);
          /* Never going to be resolved */
          goto CLEANUP;
        }
      }

    }

  } else {
    if (*status != SAI__OK) {
      *status = DAT__WEIRD;
      emsRep("dat1CvtChar_3", "Possible programming error in dat1CvtChar",
             status);
      goto CLEANUP;
    }
  }

  if ( (*nbad) > 0 ) {
    if (*status == SAI__OK) {
      *status = DAT__CONER;
      emsRep("dat1CvtChar_coner", "Some string conversions involved bad values",
             status);
    }
  }

 CLEANUP:
  if (buffer) MEM_FREE( buffer );
  return *status;
}
