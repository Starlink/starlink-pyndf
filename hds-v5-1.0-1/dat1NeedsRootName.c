/*
*+
*  Name:
*     dat1NeedsRootName

*  Purpose:
*     Retrieve the name of the root group

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     dat1NeedsRootName( hid_t objid, hdsbool_t wantprim, char * rootname,
*                        size_t rootnamelen, int * status );

*  Arguments:
*     objid = hid_t (Given)
*        Group or file identifier used to obtain the "/" group.
*     wantprim = hdsbool_t (Given)
*        If true the name of the root primitive should be returned
*        if the HDS root is not a group. If false the rootname will only
*        be filled in if the HDS root is a group.
*     rootname = char * (Given & Returned)
*        Name of the root group. Can be NULL.
*     rootnamelen = size_t (Given)
*        Allocated size of rootname.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Examines the root group to work out whether the HDS root is a
*     primitive or a structure. Depending on the value of wantprim the
*     name of the HDS root will be stored.

*  Returned Value:
*     needsroot = hdsbool_t
*        True if the HDS root is a group. False if it is a primitive.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-11-22 (TIMJ):
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

hdsbool_t
dat1NeedsRootName( hid_t objid, hdsbool_t wantprim, char * rootname, size_t rootnamelen, int * status ) {
  hdsbool_t needroot = HDS_FALSE;
  hid_t group_id;

  if (*status != SAI__OK) return needroot;

  CALLHDF( group_id,
           H5Gopen2(objid, "/", H5P_DEFAULT),
           DAT__HDF5E,
           emsRepf("hdsOpen_2","Error opening root group to get name",
                   status)
           );

  /* If the attribute indicating we have to use a primitive as the top
     level is present we open that for the root locator */
  if (H5Aexists( group_id, HDS__ATTR_ROOT_PRIMITIVE) ) {

    /* Primitive is the root locator. Retrieve the name if requested */
    if (wantprim && rootname) {
      dat1GetAttrString( group_id, HDS__ATTR_ROOT_PRIMITIVE, HDS_FALSE,
                         NULL, rootname, sizeof(rootnamelen), status );
    }
  } else {

    if (rootname) dat1GetAttrString( group_id, HDS__ATTR_ROOT_NAME, HDS_TRUE,
                                     "HDF5ROOT", rootname, rootnamelen, status );

    needroot = HDS_TRUE;
  }

 CLEANUP:
  if (group_id) H5Gclose(group_id);
  return needroot;
}
