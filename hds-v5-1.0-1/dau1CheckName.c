/*
*+
*  Name:
*     dau1CheckName

*  Purpose:
*     Check that the supplied name string conforms to the rules

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     void dau1CheckName( const char * name, int isname, char * buf, size_t buflen,
*                         int * status );

*  Arguments:
*     name = const char * (Given)
*        Name to validate
*     isname = int (Given)
*        True if this is a name string, false if it is a type string.
*     buf = char * (Returned)
*        If "name" passes validation, on exit "buf" contains
*        a normalized version of "name".
*     buflen = size_t (Given)
*        Allocated length of "buf".
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Checks the supplied string to ensure it does not have
*     any unsupported characters.
*
*     This routine validates the syntax of a 'name' specification. If
*     successful, the contents are formatted - any embedded blanks are
*     removed and all lowercase letters are converted to uppercase.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Sets status if there is a problem with the string.
*     - The name is chosen to match the HDS internal name
*       although the API differs slightly.
*     - This routine enforces the HDS rules and HDF5 may
*       still do additional checks itself.
*     - Type strings have an additional test that there is no
*       "*" in the type string. "*" is only allowed in _CHAR types.

*  History:
*     2014-08-15 (TIMJ):
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
#include <ctype.h>

#include "ems.h"
#include "sae_par.h"

#include "dat1.h"
#include "dat_err.h"
#include "dat_par.h"

void dau1CheckName( const char * name, int isname, char * buf, size_t buflen, int * status ) {

  size_t namlen = 0;   /* Length of name string */
  size_t i = 0;        /* Position in input buffer */
  size_t oposn = 0;    /* Position in output buffer */
  int errcode;         /* Error code to use for bad string */

  if (*status != SAI__OK) return;

  if (!buf) {
    *status = SAI__ERROR;
    emsRep( "", "Null pointer supplied as output buffer (possible programming error)",
            status );
    return;
  }

  /* Assign the error code */
  errcode = (isname ? DAT__NAMIN : DAT__TYPIN);

  /* Null the destination buffer */
  oposn = 0;
  buf[oposn] = '\0';

  /* Loop over input copying to output */
  namlen = strlen(name);
  for (i = 0; i < namlen; i++) {
    if (isspace(name[i])) {
      /* do not copy */
    } else if ( oposn >= buflen) {
      /* output buffer too small */
      *status = errcode;
      emsRepf("DAU_CHECK_NAME_1",
              "Invalid %s string '%s' specified; more than "
              "%zu characters long (possible programming error).",
              status,
              (isname ? "name" : "type"),
              name, buflen );
      return;
    } else if (!isprint(name[i])) {
      /* Non-printable character encountered */
      *status = errcode;
      emsRepf( "DAU_CHECK_NAME_2",
               "Invalid %s string '%s' specified; contains "
               "illegal character (code=%d decimal) at position %zu "
               "(possible programming error).",
               status,
               (isname ? "name" : "type"),
               name, (int)name[i], i);
      return;
    } else if (!isname && name[0] != '_' && name[i] == '*') {
      *status = errcode;
      emsRepf( "DAU_CHECK_NAME_4",
               "Invalid type string '%s' specified; the '*' "
               "character is not permitted in user-defined HDS types "
               "(possible programming error).",
               status, name );
      return;
    } else {
      buf[oposn] = toupper(name[i]);
      oposn++;
    }

  }

  /* Make sure we copied something -- although a blank type
     is allowed (see ary1_retyp.f) */
  if (oposn == 0 && isname) {
    *status = errcode;
    emsRepf("DAU_CHECK_NAME_3",
            "Invalid blank %s string specified "
            "(possible programming error).", status,
            (isname ? "name" : "type")
            );
    return;
  }

  /* Terminate the string */
  buf[oposn] = '\0';

}

