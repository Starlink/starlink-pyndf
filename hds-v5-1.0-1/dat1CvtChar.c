/*
*+
*  Name:
*     dat1CvtChar

*  Purpose:
*     Translate data to or from character format

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
*     character formats and is relatively inefficient.

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
dat1CvtChar( size_t nval, hdstype_t intype, size_t nbin,
             hdstype_t outtype, size_t nbout, const void * imp, void * exp,
             size_t *nbad, int * status ) {
  size_t n;
  char * buffer = NULL;
  HdsTypeInfo *typeinfo;

  *nbad = 0;
  if (*status != SAI__OK) return *status;

  /* Sanity check */
  if (intype != HDSTYPE_CHAR && outtype != HDSTYPE_CHAR) {
    *status = DAT__TYPIN;
    emsRep("dat1CvtChar", "dat1CvtChar can not do arbitrary type conversion."
           " (Possible programming error)", status );
    return *status;
  }

  /* if the types are the same and the sizes are the same,
     just copy directly. ie both strings of _CHAR*n */
  if (intype == outtype && nbin == nbout) {
    memmove( exp, imp, nval * nbin );
    return *status;
  }

  if (intype == HDSTYPE_CHAR && outtype == HDSTYPE_CHAR) {
    *status = DAT__TYPIN;
    emsRep("dat1CvtChar_2", "String to string conversion is done by HDF5"
           " (Possible programming error)", status);
    return *status;
  }

  /* Get cached type information */
  typeinfo = dat1TypeInfo();

  /* Convert each value, one at a time -- this is the case for string
     given, and non-string returned. Note that we convert EVERY element
     and set bad status if we end up having any bad elements inserted.
     We do not stop the conversion on the first bad copy. */
  if (intype == HDSTYPE_CHAR) {
    const char * inbuf = NULL;

    /* The input is in a _CHAR*nbin array and for sscanf to work
       we need a nul-terminated string */
    buffer = MEM_MALLOC( nbin + 1 );
    buffer[nbin] = '\0';
    inbuf = imp;

    for (n = 0; n < nval; n++) {
      int nitem;
      int outint;
      float outreal;
      double outdouble;
      hdsbool_t outlogical;
      short outword;
      unsigned short outuword;
      int64_t outint64;

      strncpy( buffer, inbuf, nbin );
      inbuf += nbin; /* Go to next string */

      switch( outtype ) {
      case HDSTYPE_INTEGER:
        nitem = sscanf(buffer, "%d", &outint );
        if (nitem == 0) {
          (*nbad)++;
          outint = typeinfo->BADI;
        }
        ((int *)exp)[n] = outint;
        break;
      case HDSTYPE_REAL:
        nitem = sscanf(buffer, "%f", &outreal);
        if (nitem == 0) {
          (*nbad)++;
          outreal = typeinfo->BADR;
        }
        ((float *)exp)[n] = outreal;
        break;
      case HDSTYPE_DOUBLE:
        nitem = sscanf(buffer, "%lf", &outdouble);
        if (nitem == 0) {
          (*nbad)++;
          outdouble = typeinfo->BADD;
        }
        ((double *)exp)[n] = outdouble;
        break;
      case HDSTYPE_INT64:
        nitem = sscanf(buffer, "%ld", &outint64);
        if (nitem == 0) {
          (*nbad)++;
          outint64 = typeinfo->BADK;
        }
        ((int64_t *)exp)[n] = outint64;
        break;
      case HDSTYPE_LOGICAL:
        /* could be a string TRUE/FALSE/YES/NO
           but oddly, not 1/0. HDS assumes that anything
           that is not true is always false and does not
           attempt to trap for bad values. */
        if (buffer[0] == 'T' || buffer[0] == 't' ||
            buffer[0] == 'Y' || buffer[0] == 'y' ) {
          outlogical = HDS_TRUE;
        } else {
          outlogical = HDS_FALSE;
        }
        ((hdsbool_t *)exp)[n] = outlogical;
        break;
      case HDSTYPE_BYTE:
        nitem = sscanf(buffer, "%hd", &outword);
        if ((nitem > 0) && (outword <= SCHAR_MAX) && (outword >= SCHAR_MIN)) {
          /* can just use outword as is */
        } else {
          (*nbad)++;
          outword = typeinfo->BADB;
        }
        ((char *)exp)[n] = outword;
        break;
      case HDSTYPE_UBYTE:
        nitem = sscanf(buffer, "%hd", &outword);
        if ((nitem > 0) && (outword <= UCHAR_MAX) && (outword >= 0)) {
          /* can just use outword as is */
        } else {
          (*nbad)++;
          outword = typeinfo->BADUB;
        }
        ((unsigned char *)exp)[n] = outword;
        break;
      case HDSTYPE_WORD:
        nitem = sscanf(buffer, "%hd", &outword);
        if (nitem == 0) {
          (*nbad)++;
          outword = typeinfo->BADW;
        }
        ((short *)exp)[n] = outword;
        break;
      case HDSTYPE_UWORD:
        nitem = sscanf(buffer, "%hu", &outuword);
        if (nitem == 0) {
          (*nbad)++;
          outuword = typeinfo->BADUW;
        }
        ((unsigned short *)exp)[n] = outuword;
        break;
      case HDSTYPE_CHAR:
        /* handled previously and we should not be here */
        if (*status == SAI__OK) {
          *status = DAT__WEIRD;
          emsRep("dat1CvtChar_internal",
                 "Internal consistency error on string conversion", status );
          goto CLEANUP;
        }
        break;
      default:
        if (*status == SAI__OK) {
          *status = DAT__TYPIN;
          emsRepf("dat1CvtChar_exp", "dat1CvtChar: Unsupported output data type %d",
                  status, outtype);
          /* Never going to be resolved */
          goto CLEANUP;
        }
      }
    }
  } else if (outtype == HDSTYPE_CHAR) {
    char * outbuf = NULL;
    size_t i;
    /* each value is converted one element at a time.
       We sprintf into a fixed size buffer and copy into
       the correct place in the output */
    buffer = MEM_MALLOC( nbout + 1 );
    buffer[nbout] = '\0';
    outbuf = exp;

    for (n = 0; n < nval; n++) {
      hdsbool_t inlogical;
      size_t nchar = 0;

      switch( intype ) {

      case HDSTYPE_INTEGER:
        nchar = snprintf( buffer, nbout+1, "%d", ((int *)imp)[n] );
        break;
      case HDSTYPE_REAL:
        nchar = snprintf( buffer, nbout+1, "%G", ((float *)imp)[n] );
        break;
      case HDSTYPE_DOUBLE:
        nchar = snprintf( buffer, nbout+1, "%.*G", DBL_DIG, ((double *)imp)[n] );
        break;
      case HDSTYPE_INT64:
        nchar = snprintf( buffer, nbout+1, "%ld", ((int64_t *)imp)[n] );
        break;
      case HDSTYPE_LOGICAL:
        inlogical = ((hdsbool_t *)imp)[n];
        if ( inlogical == typeinfo->BADL ) {
          buffer[0] = '*';
          buffer[1] = '\0';
          nchar = 1;
        } else if ( HDS_ISTRUE(inlogical) ) {
          /* HDS is happy to truncate FALSE to FAL if there isn't space */
          nchar = snprintf( buffer, nbout+1, "%s", "TRUE" );
        } else {
          nchar = snprintf( buffer, nbout+1, "%s", "FALSE" );
        }
        break;
      case HDSTYPE_BYTE:
        nchar = snprintf( buffer, nbout+1, "%d", ((char *)imp)[n] );
        break;
      case HDSTYPE_UBYTE:
        nchar = snprintf( buffer, nbout+1, "%u", ((unsigned char *)imp)[n] );
        break;
      case HDSTYPE_WORD:
        nchar = snprintf( buffer, nbout+1, "%d", ((short *)imp)[n] );
        break;
      case HDSTYPE_UWORD:
        nchar = snprintf( buffer, nbout+1, "%u", ((unsigned short *)imp)[n] );
        break;
      case HDSTYPE_CHAR:
        /* handled previously and we should not be here */
        if (*status == SAI__OK) {
          *status = DAT__WEIRD;
          emsRep("dat1CvtChar_internal2",
                 "Internal consistency error on string conversion", status );
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

      /* Copy the buffer to the output buffer -- space padding as this is really
         a Fortran string. */
      strncpy( outbuf, buffer, nchar );
      for (i=nchar; i<nbout; i++) {
        outbuf[i] = ' ';
      }
      outbuf += nbout;
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
