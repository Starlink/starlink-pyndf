/*
*+
*  Name:
*     dat1GetParentID

*  Purpose:
*     Returns the parent group

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     group_id = dat1GetParentID( hid_t objid, hdsbool_t allow_root, int * status ) {

*  Arguments:
*     objid = hid_t (Given)
*        Identifier of HDF5 object for which to obtain the parent group.
*     allow_root = hdsbool_t (Given)
*        Normally, HDS does not recognize the existence of the root group and datParen
*        will complain if there is no parent. Internally, "/" can be useful for some
*        routines. If true the root group can be located.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Given an HDF5 object identifier, determine the parent group and open it.

*  Returned Value:
*     hid_t parent_group_id
*       Parent group. Should be freed by calling H5Gclose. Returns negative
*       value on error.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - It is always an error to call this routine with the root group.

*  History:
*     2014-11-21 (TIMJ):
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

#include "hdf5.h"

#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

hid_t
dat1GetParentID( hid_t objid, hdsbool_t allow_root, int * status ) {
  hid_t parent_id = -1;
  ssize_t lenstr = 0;
  char * tempstr = NULL;

  if (*status != SAI__OK) return parent_id;

  /* Not sure if there is a specific API for this. For now,
     get the full name of the object and then open the group
     with the lowest part of the path removed */
  tempstr = dat1GetFullName( objid, 0, &lenstr, status );

  if (*status == SAI__OK && lenstr <= 1) {
    *status = DAT__OBJIN;
    emsRep("datParen_0",
           "Object is the HDF5 root group and has no parent "
           "group (possible programming error).", status);
    goto CLEANUP;
  }

  /* Now walk through the name in reverse and nul out the first "/"
     we encounter. */
  if (*status == SAI__OK) {
    ssize_t iposn;
    ssize_t i;
    for (i = 0; i < lenstr; i++) {
      iposn = lenstr - (i+1);
      if (tempstr[iposn] == '/') {
        tempstr[iposn] = '\0';
        break;
      }
    }
  }

  /* if this seems to be the root group we rewrite it to be "/" else,
     optionally return an error. */
  if (tempstr[0] == '\0') {
    if (allow_root) {
      tempstr[0] = '/';
      tempstr[1] = '\0';
    } else if (*status == SAI__OK) {
      *status = DAT__OBJIN;
      emsRep("datParen_1",
             "Object is a top-level object and has no parent "
             "structure (possible programming error).", status);
      goto CLEANUP;
    }
  }

  /* It seems you can open a group on an arbitrary
     item (group or dataset) if you use a fully specified
     path. This means you do not need to get an
     explicit file_id to open the group */
  CALLHDF(parent_id,
          H5Gopen(objid, tempstr, H5P_DEFAULT),
          DAT__HDF5E,
          emsRepf("datParen_2", "Error opening parent structure '%s'",
                  status, tempstr );
          );

 CLEANUP:
  if (tempstr) MEM_FREE(tempstr);
  return parent_id;
}
