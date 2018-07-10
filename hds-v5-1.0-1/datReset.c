/*
*+
*  Name:
*     datReset

*  Purpose:
*     Reset object state

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datReset(const HDSLoc *locator, int *status);

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Primitive locator.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Reset the state of a primitive, ie. "un-define" its value. All
*     subsequent read operations will fail until the object is written
*     to (re-defined).

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - All data are deleted from the primitive.

*  History:
*     2014-10-16 (TIMJ):
*        Initial version
*     2014-11-21 (TIMJ):
*        Stop using an attribute and switch to deleting
*        the primitive and recreating it empty.
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

int
datReset(HDSLoc *locator, int *status) {
  unsigned intent = 0;
  char name_str[DAT__SZNAM+1];
  hid_t h5type = -1;
  hid_t new_dataset_id = -1;
  hid_t new_dataspace_id = -1;
  hid_t parent_id = -1;
  hsize_t h5dims[DAT__MXDIM];
  int rank;

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datReset", 1, locator, 0, status );
  datName( locator, name_str, status );

  if (dat1IsStructure(locator, status)) {
    *status = DAT__OBJIN;
    emsRepf("datState_1", "datReset: '%s' is not a primitive locator",
            status, name_str);
    return *status;
  }

    /* How did we open this file? */
  CALLHDFQ( H5Fget_intent( locator->file_id, &intent ));
  /* Must check whether the file was opened for write */
  if ( intent == H5F_ACC_RDONLY ) {
    *status = DAT__ACCON;
    emsRepf("datReset", "datReset: Can not reset readonly primitive",
            status);
    goto CLEANUP;
  }

  /* Delete, and recreate empty with the same type and dims. Need the parent
     to delete and recreate. */
  parent_id = dat1GetParentID( locator->dataset_id, HDS_TRUE, status );

  /* Dimensions for the new dataspace */
  CALLHDFE( int,
            rank,
            H5Sget_simple_extent_dims( locator->dataspace_id, h5dims, NULL ),
            DAT__DIMIN,
            emsRep("datReset_dims", "datReset: Error obtaining shape of object",
                   status)
            );

  /* Data type that we need */
  CALLHDF( h5type,
           H5Dget_type( locator->dataset_id ),
           DAT__HDF5E,
           emsRep("dat1Type_1", "datType: Error obtaining data type of dataset", status)
           );

  /* Delete the current dataset */
  CALLHDFQ( H5Ldelete( parent_id, name_str, H5P_DEFAULT ));

  /* Create the brand new primitive */
  /* Create the brand new primitive */
  dat1NewPrim( parent_id, rank, h5dims, h5type, name_str, &new_dataset_id,
               &new_dataspace_id, status );

  if (*status == SAI__OK) {
    H5Sclose(locator->dataspace_id);
    locator->dataspace_id = new_dataspace_id;
    H5Dclose(locator->dataset_id);
    locator->dataset_id = new_dataset_id;
  }

 CLEANUP:
  if (h5type > 0) H5Tclose(h5type);
  if (parent_id > 0) H5Gclose(parent_id);
  if (*status != SAI__OK) {
    if (new_dataspace_id > 0) H5Sclose(new_dataspace_id);
    if (new_dataset_id > 0) H5Dclose(new_dataset_id);
  }
  return *status;
}
