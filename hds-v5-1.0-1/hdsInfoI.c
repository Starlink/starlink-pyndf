/*
*+
*  Name:
*     hdsInfoI

*  Purpose:
*     Retrieve internal state from HDS as integer

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hdsInfoI(const HDSLoc* loc, const char * topic, const char * extra,
*              int *result, int * status);

*  Arguments:
*     loc = const HDSLoc* (Given)
*        HDS locator, if required by the particular topic. Will be
*        ignored for FILES and LOCATORS topics and can be NULL pointer.
*     topic = const char * (Given)
*        Topic on which information is to be obtained. Allowed values are:
*        - LOCATORS : Return the number of active locators.
*                     Internal root scratch locators are ignored.
*        - ALOCATORS: Returns the number of all active locators, including
*                     scratch space.
*        - FILES : Return the number of open files
*        - VERSION : Return the HDS implementation version number for the
*                    supplied HDS locator.
*     extra = const char * (Given)
*        Extra options to control behaviour. The content depends on
*        the particular TOPIC. See NOTES for more information.
*     result = int* (Returned)
*        Answer to the question.
*     status = int* (Given & Returned)
*        Variable holding the status value. If this variable is not
*        SAI__OK on input, the routine will return without action.
*        If the routine fails to complete, this variable will be
*        set to an appropriate error number.

*  Description:
*     Retrieves integer information associated with the current state
*     of the HDS internals.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     DSB:  David S Berry (EAO):
*     {enter_new_authors_here}

*  Notes:
*     - Can be used to help debug locator leaks.
*     - The "extra" information is used by the following topics:
*       - "LOCATORS", if non-NULL, "extra" can contain a comma
*         separated list of locator paths (upper case, as returned
*         by hdsTrace) that should be included in the count. If any
*         component is preceeded by a '!' all locators starting
*         with that path will be ignored in the count. This can be
*         used to remove parameter locators from the count.
*       - If "!EXTINCTION,EXTINCTION" is requested then they will
*         match everything, since the test is performed on each
*         component separately.
*       - Only valid hds locators are counted. If there is an internal
*         error tracing a locator, it is ignored and that locator is
*         not included in the count.
*    - Top-level scratch locators such as "HDS_SCRATCH.TEMP_N" are not
*      included in the "LOCATORS" count but children of the temp locators
*      are included (since those temporary items should be freed).

*  History:
*     2014-10-17 (TIMJ):
*        Initial version
*     2017-05-16 (DSB):
*        Add topic VERSION.
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

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

/* Maximum number of filters that we can supply */
#define MAXCOMP 20

int
hdsInfoI(const HDSLoc* loc, const char *topic_str, const char *extra,
	 int *result, int  *status) {

  if (*status != SAI__OK) return *status;

  if (strncasecmp(topic_str, "VERSION", 7) == 0) {
    *result = loc->hds_version;
  } else if (strncasecmp(topic_str, "FIL", 3) == 0) {
    *result = hds1CountFiles();
  } else if (strncasecmp(topic_str, "ALOC", 4) == 0 ||
             strncasecmp(topic_str, "LOCA", 4) == 0 ) {
    char * filter = NULL;
    char *comps[MAXCOMP];
    size_t ncomp = 0;
    hdsbool_t skip_scratch_root = HDS_FALSE;

    if (topic_str[0] == 'L') skip_scratch_root = HDS_TRUE;

    if (extra) {
      size_t j = 0;
      size_t i = 0;
      size_t len = strlen(extra);
      int atstart = 1; /* indicates that the next time round is the start */

      /* Get some memory for the fixed up filter list. */
      filter = MEM_MALLOC( (len+1) * sizeof(*filter) );

      /* Copy characters from the EXTRA input to the output,
         uppercasing as we go and removing spaces. We look for
         commas and store the positions of the character after the
         comma in the char** array. */
      for (i=0; i<len; i++) {
        if ( extra[i] != ' ') {
          if (extra[j] == ',') {
            /* store a string terminator and indicate that next time
               around we store the pointer */
            filter[j] = '\0';
            atstart = 1;
          } else {
            filter[j] = toupper(extra[i]);
            if (atstart) {
              comps[ncomp] = &(filter[j]);
              atstart = 0;
              ncomp++;
              if (ncomp >= MAXCOMP) {
                *status = DAT__NOMEM;
                emsSeti("MAX", MAXCOMP);
                emsRep("HDSINFOI",
                       "Too many components to filter on. Max = ^MAX",
                       status);
                break;
              }
            }
          }
          j++;
        }
      }
      filter[j] = '\0';
    }
    *result = hds1CountLocators( ncomp, comps, skip_scratch_root, status );
    MEM_FREE(filter);
  }

  return *status;
}
