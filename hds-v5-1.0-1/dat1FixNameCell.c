/*
*+
*  Name:
*     dat1FixNameCell

*  Purpose:
*     Correct path of component to correct for arrays of structures

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     char * dat1FixNameCell( const char * instr, int * status );

*  Arguments:
*     inst = const char * (Given)
*        String to correct.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Returned Value:
*     char * = corrected string
*        If the string instr contained references to arrays of structures
*        a newly allocated buffer will be returned with the array parts
*        of the path adjusted. Returns NULL if the input string does
*        not need to be modified. The returned buffer should be freed
*        with MEM_FREE.

*  Description:
*    Any path in a HDS tree may include a group array reference
*    that we need to remove before we scan back looking for the
*    path separator. e.g HISTORY.RECORD.HDSCELL(5) should
*    be returned as name RECORD(5). This routine looks for
*    the relevant cell specification text, and if necessary
*    allocates a new buffer and copies in a version with the
*    HDF5 implementation details removed.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Does not convert "/" to "." and assumes that the supplied
*       path is an HDF5 path and not an HDS path.

*  History:
*     2014-09-06 (TIMJ):
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

#include "hds1.h"
#include "dat1.h"
#include "hds.h"


char * dat1FixNameCell( const char * instr, int * status ) {

  char * cleanstr = NULL;
  char * cellstr = NULL;
  ssize_t lenstr = 0;

  /* We are looking for "/HDSCELL(" so that we can be
     sure things are not fluking a name of HDSCELL
     (or whatever value is now in DAT__CELLNAME) */
  const char cellroot[] = "/" DAT__CELLNAME "(";

  if (*status != SAI__OK) return NULL;

  lenstr = strlen(instr);

  cellstr = strstr( instr, cellroot );

  /* if we did not find anything we do nothing and just immediately
     return */
  if (cellstr) {
    ssize_t i;
    size_t oposn = 0;
    size_t rootlen = strlen(cellroot);
    /* need to copy the characters over, missing out the
       cell name (plus the ".") */
    cleanstr = MEM_MALLOC( lenstr + 1 );

    for (i=0; i <= lenstr; i++) {
      if ( cellstr == &(instr[i]) ) {
        /* start of an HDSCELL string so skip that
           many characters -- we add the "(" */
        i += rootlen - 1; /* remember rootlen has the "(" and will be incremented again */
        cleanstr[oposn] = '(';

        /* Look for more HDSCELL entries */
        cellstr = strstr( &(instr[i]), cellroot );

      } else {
        cleanstr[oposn] = instr[i];
      }
      oposn++;
    }

  }

  return cleanstr;

}
