/*
*+
*  Name:
*     hdsTrace

*  Purpose:
*     Trace object path

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hdsTrace(const HDSLoc *locator, int  *nlev, char *path_str,
*              char *file_str, int  *status, size_t path_length,
*              size_t file_length);

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Object locator.
*     nlev = *int (Returned)
*        Number of path levels.
*     path_str = char * (Returned)
*        Object path name within container file.
*     file_str = char * (Returned)
*        Container file name.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Trace the path of an object and return the fully resolved name as a text string.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     The path name is constructed as a sequence of "node"
*     specifications thus: node.node......node.object such as:
*     NGC1365.SKYPOS.RA.MINUTES where MINUTES is the name of the
*     object associated with the specified locator and NGC1365 is the
*     top-level object in the structure. If any of the nodes are
*     non-scalar the appropriate subscript expression is included
*     thus: AAO.OBS(6).IMAGE_DATA If the bottom-level object is a
*     slice or cell of an array, the appropriate subscript expression
*     is appended thus: M87.MAP(100:412,200:312) or CENA(3,2)

*  History:
*     2014-09-02 (TIMJ):
*        Initial version
*     2014-11-22 (TIMJ):
*        The HDS root group may not be in the HDF5 full name
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

#include "star/one.h"

#include "hdf5.h"

#include "ems.h"
#include "sae_par.h"
#include "prm_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

static void objid_to_name ( hid_t objid, int asfile, char * buffer, size_t buflen,
                            int *status);

int hdsTrace(const HDSLoc *locator, int  *nlev, char *path_str,
             char *file_str, int  *status, size_t path_length,
             size_t file_length) {

  hid_t objid = 0;

  *nlev = 0;
  if (*status != SAI__OK) return *status;

  /* don't care whether this is group or dataset */
  objid = dat1RetrieveIdentifier( locator, status );
  if (*status != SAI__OK) return *status;

  /* First we get the path of the object */
  objid_to_name( objid, 0, path_str, path_length, status );

  /* Now walk through the string replacing "/" with "." */
  if (*status == SAI__OK) {
    hdsdim lower[DAT__MXDIM];
    hdsdim upper[DAT__MXDIM];
    char subscriptstr[2 + 2 * (VAL__SZK + 2 ) * DAT__MXDIM + 1]; /* "(a1:a2,b,c,d,...)" */
    int rank;
    size_t i;
    hdsbool_t issubset = 0;
    size_t lenstr = strlen(path_str);
    for (i = 0; i < lenstr; i++) {
      if ( path_str[i] == '/' ) {
        path_str[i] = '.';
        (*nlev)++;
      }
    }
    /* the level is one more than the number of dots we found
       (assuming that objid_to_name did not return the
       root ".") */
    (*nlev)++;

    /* if this is a slice or a cell of a primitive object then
       we need to include that information in the full path name */
    dat1GetBounds( locator, lower, upper, &issubset, &rank, status );

    if (issubset) {
      dat1EncodeSubscript( rank, 1, lower, upper, subscriptstr, sizeof(subscriptstr), status );
      one_strlcat( path_str, subscriptstr, path_length, status );
    }

  }

  /* Now the file name */
  objid_to_name( objid, 1, file_str, file_length, status );

  return *status;
}


static void objid_to_name ( hid_t objid, int asfile, char * buffer, size_t buflen,
                            int *status) {
  char *tempstr = NULL;
  char *cleanstr = NULL;
  size_t iposn = 0;
  hdsbool_t needroot = HDS_FALSE;

  if (*status != SAI__OK) return;

  tempstr = dat1GetFullName( objid, asfile, NULL, status );

  /* Handle the presence of a HDF5 array structure path
     that needs to be converted to HDS hierarchy */
  if (!asfile) cleanstr = dat1FixNameCell( tempstr, status );
  if (!cleanstr) cleanstr = tempstr;

  /* For paths we need to account for the root group "/" not
     having a name in HDF5 but having a name in HDS */
  if (!asfile) {
    needroot = dat1NeedsRootName( objid, HDS_FALSE, buffer, buflen, status );
    }

  /* and copy it into the supplied buffer.
     For paths we sometimes start at the second character as we do not
     always want the leading "." (aka "/" root).
   */
  iposn = (asfile ? 0 : (needroot ? 0 : 1));

  if (needroot) {
    if (strlen( &cleanstr[iposn] ) > 1) one_strlcat( buffer, &(cleanstr[iposn]), buflen, status );
  } else {
    one_strlcpy( buffer, &(cleanstr[iposn]), buflen, status );
  }
  if (cleanstr != tempstr) MEM_FREE(cleanstr);
  if (tempstr) MEM_FREE(tempstr);

  return;
}
