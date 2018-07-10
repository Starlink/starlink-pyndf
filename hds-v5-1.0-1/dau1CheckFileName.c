/*
*+
*  Name:
*     dau1CheckFileName

*  Purpose:
*     Return checked version of file name with appropriate extension

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     fname = dau1CheckFileName( const char * file_str,  int * status );

*  Arguments:
*     file_str = const char * (Given)
*        File name supplied by user.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Returned Value:
*     fname = char *
*        Updated filename with shall variables expended and with extension
*        added if appropriate. Should be freed by calling MEM_FREE.

*  Description:
*     Validate the supplied file name and add the standard extension if
*     required. If a shell has been enabled via the SHELL tuning parameter
*     wildcards and shell variables will be expanded.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Returned pointer must be freed with MEM_FREE

*  History:
*     2014-08-29 (TIMJ):
*        Initial version
*     2014-11-18 (TIMJ):
*        Add shell expansion
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

#include <stdlib.h>
#include <string.h>
#include <wordexp.h>
#include <ctype.h>

#include "hdf5.h"

#include "star/util.h"
#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

#if __MINGW32__
      /* Use Windows separator */
#define DIRSEP  '\\'
#else
#define DIRSEP  '/'
#endif

char *
dau1CheckFileName( const char * file_str, int * status ) {

  ssize_t lenstr;
  char *fname = NULL;
  int needext = 0;
  ssize_t i;
  ssize_t endpos;
  ssize_t startpos = 0;
  ssize_t dotpos = -1;
  ssize_t dirpos = -1;
  hdsbool_t special = HDS_FALSE;
  size_t outstrlen = 0;
  size_t nspaces = 0;   /* Spaces in path */

  if (*status != SAI__OK) return NULL;

  /* HDSv4 removes leading and trailing whitespace */
  lenstr = strlen(file_str);
  endpos = lenstr;

  /* Check for trailing whitespace */
  for ( ; lenstr > 0; lenstr-- ) {
    if ( !isspace( file_str[lenstr-1] ) ) {
      endpos = lenstr;
      break;
    }
  }

  /* Find where non-whitespace starts */
  for ( ; startpos < lenstr; startpos++ ) {
    if ( !isspace( file_str[startpos] ) ) break;
  }

  /* blank file is not good */
  if ( startpos == endpos ) {
    *status = DAT__FILNF;
    emsRep("dau1CheckFileName_1", "Invalid blank file name given",
           status );
    goto CLEANUP;
  }

  /* Now scan through the file name recording the position of the last
     slash, the last dot and whether there are any special characters
     that might require shell expansion. */
  for (i=startpos; i<lenstr; i++) {
    switch ( file_str[i] ) {
    case '.':
      dotpos = i;
      break;

    case ' ':
      nspaces++;
      break;

    case DIRSEP:
      dirpos = i;
      break;

      /* _ and - are allowed in portable file names so no special action */
    case '_':
    case '-':
      break;

    default:
      /* all other characters are assumed to be restricted to shell metacharacters */
      if ( !isalnum( file_str[i] ) ) special = HDS_TRUE;
    }
  }

  /* We only need to add a file extension if the dot comes before the
     directory separator */
  needext = ( dotpos <= dirpos );

  /* only need to worry about special characters if we have found some
     and if the tuning parameter indicates we can use a shell expansion */
  if ( hds1GetShell() == HDS__NOSHELL ) special = 0;

  /* Work out length of buffer required without shell expansion (include terminator) */
  outstrlen = ( lenstr - startpos ) + 1 + 1;
  if (needext) outstrlen += DAT__SZFLX;

  /* We will need a buffer filled with the input string
     if there are no special characters (because we just return it)
     of if we need the extension added */

  if ( !special || needext ) {

    fname = MEM_MALLOC( outstrlen );
    if (!fname) {
      *status = DAT__NOMEM;
      emsRep("", "Error in a string malloc. This is not good",
             status );
      goto CLEANUP;
    }

    /* Do not need error checking version as we know the length of the input
       might be too long (trailing spaces) but it does not matter as we
       also know it will fit. */
    star_strlcpy( fname, &(file_str[startpos]), outstrlen );

    /* NUL terminate so trailing spaces are ignored. We could not
       terminate file_str because it is const */
    fname[lenstr-startpos] = '\0';

    /* Append the file extension if necessary */
    if (needext) star_strlcat( fname, DAT__FLEXT, outstrlen );

  }

  /* Shell expansion here, using either the temporary buffer or the input buffer.
     We use the wordexp() system call. */
  if (special) {
    int retval = 0;      /* Status from wordexp() */
    wordexp_t pwordexp;  /* Results from wordexp */
    const char * tmpbuffer;

    tmpbuffer = fname ? fname : &(file_str[startpos]);
    retval = wordexp( tmpbuffer, &pwordexp, 0 );

    if (retval == 0) {

      if (pwordexp.we_wordc == 1) {
        /* one match so we will copy that into an output buffer */
        size_t szword;
        tmpbuffer = (pwordexp.we_wordv)[0];
        szword = strlen( tmpbuffer ) + 1;

        /* Grow the buffer as required [fname may or may not be NULL already] */
        fname = MEM_REALLOC( fname, szword );
        star_strlcpy( fname, tmpbuffer, szword );

      } else if (pwordexp.we_wordc > 1) {
        *status = DAT__FATAL;
        emsRepf("dau1CheckFileName_toomany",
                "%d results from string expansion of '%s'"
                " but HDS can only open a single file at a time",
                status, (int)(pwordexp.we_wordc), tmpbuffer );
        goto CLEANUP;
      } else {
        /* This should not happen as wordexp() always seems to return something
           even if it is a copy of the input */
        *status = DAT__FILNF;
        emsRepf("dau1CheckFileName_worexp_c",
                "Shell expansion failed to find any results", status );
        goto CLEANUP;
      }
    } else {
      *status = DAT__FATAL;
      emsRepf("dau1CheckFileName_wordexp", "Internal error (%d) from wordexp()",
              status, retval );
      goto CLEANUP;
    }
  }

 CLEANUP:
  if (*status != SAI__OK) {
    if (fname) {
      MEM_FREE(fname);
      fname = NULL;
    }
  }
  return fname;
}
