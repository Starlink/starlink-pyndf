/*
*+
*  Name:
*     dau1CheckType

*  Purpose:
*     Check HDS type and convert to HDF5 data type

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     int dau1CheckType( hdsbool_t asmem, const char * type_str, hid_t *h5type,
*                        char * norm_str, size_t normlen, int * status );

*  Arguments:
*     asmem = hdsbool_t (Given)
*        If true (1) the resulting type should be the type
*        to be used for in-memory operations. If false (0)
*        the type returned should be the type used in the
*        HDF5 file.
*     type_str = const char * (Given)
*        HDS data type.
*     h5type = hid_t * (Returned)
*        Data type to use if the supplied type_str looks like a
*        primitive type. Not modified if it seems to be
*        referring to a structure. See notes for details.
*        Types are always copies and should be freed with H5Tclose.
*     norm_str = char * (Given and Returned)
*        Normalized form of the supplied type string. Will contain
*        the upper-cased version of the type_str with spaces
*        removed. This will be the required "group" name if the
*        supplied type specified a non-primitive type.
*     normlen = size_t (Given)
*        Allocated size of norm_str.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Returned Value:
*     int = True if the data type was a primitive type, false
*           if it was a structure.

*  Description:
*     Converts the numeric HDS data types to HDF5 equivalent
*     data types. Sets status to bad if a numeric type (one with
*     a leading underscore) is not recognized. All other types
*     are assumed to be group names.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     Mapping of HDS numeric types is as follows:
*     - _BYTE : H5T_NATIVE_INT8
*     - _UBYTE : H5T_NATIVE_UINT8
*     - _WORD : H5T_NATIVE_INT16
*     - _UWORD : H5T_NATIVE_UINT16
*     - _INTEGER : H5T_NATIVE_INT32
*     - _INT64 : H5T_NATIVE_INT64
*     - _REAL : H5T_NATIVE_FLOAT
*     - _DOUBLE : H5T_NATIVE_DOUBLE
*     - _LOGICAL : H5T_NATIVE_B8 (H5T_NATIVE_B32 in memory)
*     - _CHAR*N : H5T_STRING (space padded)

*  History:
*     2014-08-18 (TIMJ):
*        Initial version
*     2018-04-09 (DSB):
*        If the supplied type string is blank (which is allowed in HDS V4) 
*        return a normalised type consisting of a single space rather than
*        a null (i.e. zero-length) string. Using a null string causes problems 
*        when storing the type as an HDF5 attribute.  
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

#include "sae_par.h"
#include "dat1.h"
#include "hdf5.h"
#include "dat_err.h"
#include "ems.h"

#include <string.h>
#include <stdlib.h>

int dau1CheckType ( hdsbool_t asmem, const char * type_str, hid_t * h5type,
                    char * norm_str, size_t normlen, int * status ) {

  hid_t ltype = 0;

  if (*status != SAI__OK) return 1;

  /* Copy string from input to output, upper-casing
     and removing spaces. Have to do this first in case
     the supplied type starts with white space. */
  dau1CheckName( type_str, 0, norm_str, normlen, status );

  /* If the type does not begin with an _ then we assume this is
     a structure / group name. */
  if ( norm_str[0] != '_' ) {

    /* HDS V4 allows structure types to be blank. But a blank name ends
       up with zero length after removal of spaces, and returning a null
       (i.e. zero length) type string here causes problems when storing
       the type string as an HDF5 attribute (attribute names cannot be
       zero length). So instead return a single space. */
    if( strlen( norm_str ) == 0 && normlen > 1 ) strcpy( norm_str, " " );

    return 0;
  }

  /* Now check the primitive data type. There is overhead to strncmp()
     so we try to put the more common items first and realise that
     W, B, L, D and R types can match on a single character.
     _INTEGER and _INT64 clash along with _UBYTE and _UWORD.
     We do a full match on _CHAR* so that we can be sure what we have.
   */
  if ( strncmp( norm_str, "_INTE", 5 ) == 0) {
    ltype = H5T_NATIVE_INT32;
  } else if ( norm_str[1] == 'D') {
    ltype = H5T_NATIVE_DOUBLE;
  } else if ( norm_str[1] == 'R') {
    ltype = H5T_NATIVE_FLOAT;
  } else if ( norm_str[1] == 'B') {
    ltype = H5T_NATIVE_INT8;
  } else if ( norm_str[1] == 'W') {
    ltype = H5T_NATIVE_INT16;
  } else if ( norm_str[1] == 'L') {
    ltype = (asmem ? H5T_NATIVE_B32 : H5T_NATIVE_B8);
  } else if ( strncmp( norm_str, "_INT6", 5 ) == 0 ) {
    ltype = H5T_NATIVE_INT64;
  } else if ( strncmp( norm_str, "_UW", 3 ) == 0 ) {
    ltype = H5T_NATIVE_UINT16;
  } else if ( strncmp( norm_str, "_UB", 3 ) == 0 ) {
    ltype = H5T_NATIVE_UINT8;
  } else if ( strncmp( norm_str, "_CHAR", 5 ) == 0 ) {
    size_t clen = 1;

    /* If a character type specification contains no length expression, then    */
    /* its length defaults to 1.                                                */
    if (strlen(norm_str) == 5) {
      clen = 1;
    } else if ( norm_str[5] != '*' ) {
      /* If it is followed by anything except '*', then report an error.          */
      *status = DAT__TYPIN;
      emsRepf( "DAT1_CHECK_TYPE_1",
               "Invalid length encountered in the character "
               "type specification '%s'; should be '_CHAR*n' "
               "(possible programming error).",
               status, norm_str );
    } else {
      /* Read the integer following the _CHAR* specifier */

      clen = strtol(&(norm_str[6]), NULL, 10 );
      if ( clen < 1 || clen > DAT__MXCHR ) {
        *status = DAT__TYPIN;
        emsRepf( "DAT1_CHECK_TYPE_2",
                 "Invalid length encountered in the character "
                 "type specification '%s'; should be in the range 1 to %d "
                 "(possible programming error).",
                 status, norm_str, DAT__MXCHR );
        return 1;
      }

    }

    /* Need an array of characters */
    *h5type = H5Tcopy(H5T_C_S1);
    H5Tset_size( *h5type, clen );

    /* Try padding them Fortran style for now -- matching HDS */
    H5Tset_strpad( *h5type, H5T_STR_SPACEPAD );

  }

  /* Copy the types if we haven't already done it */
  if (ltype > 0) *h5type = H5Tcopy( ltype );

  return 1;
}
