/*
*+
*  Name:
*     datFind

*  Purpose:
*     Find named component

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datFind( const HDSLoc * loc1, const char *name, HDSLoc **loc2, int * status );

*  Arguments:
*     loc1 = const HDSLoc * (Given)
*        Structure locator.
*     name = const char * (Given)
*        Component name.
*     loc2 = HDSLoc ** (Returned)
*        Component locator.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*      Obtain a locator for a named component.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     If the structure is an array, loc1 must be explicitly associated
*     with an individual cell.  If the component is a structure array,
*     loc2 will be associated with the complete array, not the first
*     cell.  Access to its components can only be made through another
*     locator explicitly associated with an individual cell (see datCell).
*     If the parent locator is associated with a group, the child locator
*     will also be associated with that group.

*  History:
*     2014-08-26 (TIMJ):
*        Initial version
*     2014-11-14 (TIMJ):
*        Child locators must inherit group
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

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"
#include "sae_par.h"

int
datFind( const HDSLoc   *locator1,
         const char     *name_str,
         HDSLoc   **locator2,
         int      *status ) {

  char cleanname[DAT__SZNAM+1];
  HDSLoc * thisloc = NULL;
  H5O_info_t object_info;
  int rdonly;
  int there = 0;
  int havegroup = 0;
  int lockinfo;

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datFind", 1, locator1, 1, status );
  if (*status != SAI__OK) return *status;

  /* containing locator must refer to a group */
  if (!dat1IsStructure( locator1, status) ) {
    *status = DAT__OBJIN;
    emsRep("datFind_1", "Input object is not a structure",
           status);
    return *status;
  }

  /* Normalize the name string */
  dau1CheckName( name_str, 1, cleanname, sizeof(cleanname), status );
  if (*status != SAI__OK) return *status;

  /* First ensure that the component exists */
  datThere( locator1, cleanname, &there, status );
  if (!there) {
    if (*status == SAI__OK) {
      *status = DAT__OBJNF;
      emsRepf("datFind_1b", "datFind: Object '%s' not found",
              status, cleanname);
    }
    return *status;
  }

  /* Check the type of the requested object */
  CALLHDFQ( H5Oget_info_by_name( locator1->group_id, cleanname,
                                 &object_info, H5P_DEFAULT ) );

  /* Make sure we have a supported type */
  if (*status == SAI__OK) {
    H5O_type_t type = object_info.type;

    if (type == H5O_TYPE_GROUP) {
      havegroup = 1;
    } else if (type == H5O_TYPE_DATASET) {
      havegroup = 0;
    } else {
      *status = DAT__OBJIN;
      emsRepf("datFind_1c", "datFind: Component '%s' exists but is neither group"
              " nor dataset.", status, cleanname);
      goto CLEANUP;
    }
  }

  /* Create the locator */
  thisloc = dat1AllocLoc( status );
  if (*status != SAI__OK) goto CLEANUP;

  /* Child locators are not primary by default -- just store the file_id and register  */
  thisloc->file_id = locator1->file_id;
  hds1RegLocator( thisloc, status );

  /* Use the simplification layer to see if we have a dataset of this name */
  if (!havegroup) {
    /* we are finding a dataset */
    hid_t dataset_id;
    hid_t dataspace_id;

    CALLHDF(dataset_id,
            H5Dopen2( locator1->group_id, cleanname, H5P_DEFAULT),
            DAT__OBJIN,
            emsRepf("datFind_2", "Error opening primitive named %s", status, cleanname)
            );

    /* and data space */
    CALLHDF(dataspace_id,
            H5Dget_space( dataset_id ),
            DAT__OBJIN,
            emsRepf("datFind_2b", "Error retrieving data space from primitive named %s",
                    status, cleanname)
            );

    if (*status == SAI__OK) {
      thisloc->dataset_id = dataset_id;
      thisloc->dataspace_id = dataspace_id;
    }
  } else {
    /* Try to open it as a group */
    hid_t group_id;
    CALLHDF(group_id,
            H5Gopen2( locator1->group_id, cleanname, H5P_DEFAULT ),
            DAT__OBJIN,
            emsRepf("datFind_3", "Error opening component %s", status, cleanname)
            );
    if (*status == SAI__OK) thisloc->group_id = group_id;
  }

  /* Store a pointer to the handle for the returned HDF object */
  thisloc->handle = dat1Handle( locator1, cleanname, 0, status );

  /* We have to propagate groupness to the child */
  if ( (locator1->grpname)[0] != '\0') hdsLink(thisloc, locator1->grpname, status);

  /* Determine if the current thread has a read-only or read-write lock
     on the parent object, referenced by the supplied locator. */
  dat1HandleLock( locator1->handle, 1, 0, 0, &lockinfo, status );
  rdonly = ( lockinfo == 3 );


  /* Attempt to lock the component object for use by the current thread,
     using the same sort of lock (read-only or read-write) as the
     parent object. Report an error if this fails. */

  dat1HandleLock( thisloc->handle, 2, 0, rdonly, &lockinfo, status );
  if( !lockinfo && *status == SAI__OK ) {
     *status = DAT__THREAD;
     emsSetc( "C", name_str );
     emsSetc( "A", rdonly ? "read-only" : "read-write" );
     datMsg( "O", locator1 );
     emsRep( "","datFind: requested component ('^C') within HDS object '^O' "
             "cannot be locked for ^A access - another thread already has "
             "a conflicting lock on the same component.", status );
  }

  if (*status != SAI__OK) goto CLEANUP;
  *locator2 = thisloc;
  return *status;

 CLEANUP:
  if (thisloc) datAnnul( &thisloc, status);
  return *status;
}
