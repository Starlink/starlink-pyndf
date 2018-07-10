/*
*+
*  Name:
*     dat1NewPrim

*  Purpose:
*     Create new dataset

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     dat1NewPrim( hid_t group_id, int ndim, const hsize_t h5dims[], hid_t h5type,
                   const char * name_str, hid_t * dataset_id, hid_t *dataspace_id, int *status);

*  Arguments:
*     group_id = hid_t (Given)
*        Location in HDF5 file to receive the new dataset.
*     ndim = int (Given)
*        Number of dimensions of dataset (0 means scalar).
*     h5dims = const hsize_t [] (Given)
*        Dimensions in HDF5 C order. No more than DAT__MXDIM.
*     h5type = hid_t (Given)
*        HDF5 datatype of new dataset.
*     name_str = const char * (Given)
*        Name of new dataset. Not constrained by HDS rules so can
*        be longer than DAT__SZNAM.
*     dataset_id = hid_t * (Returned)
*        Dataset identifier of new dataset.
*     dataspace_id = hid_t * (Returned)
*        Dataspace identifier of new dataspace.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Creates an HDF5 dataset given HDF5-style arguments.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-11-05 (TIMJ):
*        Initial version. Refactored from dat1New
*     2014-11-05 (TIMJ):
*        Turn off the ability to resize datasets.
*        Rely on datAlter to not attempt this.
*     2014-11-06 (TIMJ):
*        Chunked datasets is now a compile-time switch.
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

void dat1NewPrim( hid_t group_id, int ndim, const hsize_t h5dims[], hid_t h5type,
                  const char * name_str, hid_t * dataset_id, hid_t *dataspace_id, int *status ) {
  hid_t cparms = H5P_DEFAULT;
  *dataset_id = 0;
  *dataspace_id = 0;

  if (*status != SAI__OK) return;

  if (ndim == 0) {

    CALLHDF( *dataspace_id,
             H5Screate( H5S_SCALAR ),
             DAT__HDF5E,
             emsRepf("dat1New_0", "Error allocating data space for scalar %s",
                     status, name_str )
             );

  } else {
    /* Since HDS can not tell us the largest size that the user will need for this
       dataset, if we are to allow resizing we have to make it unlimited. */
    const hsize_t *maxdims = NULL;

    CALLHDF( cparms,
             H5Pcreate( H5P_DATASET_CREATE ),
             DAT__HDF5E,
             emsRepf("dat1New_1b", "Error creating parameters for data set %s",
                     status, name_str)
             );

    /* Create a primitive -- if we create it chunked we can not memory map
       but we can resize. If we create a fixed size then in theory we can
       memory map but resizes (datAlter) have to be done by copy and delete. */

#if HDS_USE_CHUNKED_DATASETS
    /* Create the dataspace with chunked storage that is resizable. For this
       to happen we just need two updates:
       - a parameter indicating that chunking is enabled.
       - the max dimensions.
    */
    const hsize_t h5max[DAT__MXDIM] = { H5S_UNLIMITED, H5S_UNLIMITED, H5S_UNLIMITED,
                                        H5S_UNLIMITED, H5S_UNLIMITED, H5S_UNLIMITED,
                                        H5S_UNLIMITED };
    /* Unlimited dimensions */
    maxdims = h5max;

    /* We can not find out the optimum chunk size from HDS API so we choose
       the initial size. */
    CALLHDFQ( H5Pset_chunk( cparms, ndim, h5dims ) );

#endif

    /* Create the data space for the dataset */
    CALLHDF( *dataspace_id,
             H5Screate_simple( ndim, h5dims, maxdims ),
             DAT__HDF5E,
             emsRepf("dat1New_1", "Error allocating data space for %s",
                     status, name_str )
             );

  }

  /* now place the dataset */
  CALLHDF( *dataset_id,
           H5Dcreate2(group_id, name_str, h5type, *dataspace_id,
                      H5P_DEFAULT, cparms, H5P_DEFAULT),
           DAT__HDF5E,
           emsRepf("dat1New_2", "Error placing the data space in the file for %s",
                   status, name_str )
           );

 CLEANUP:
  if (*status != SAI__OK) {
    /* tidy */
    if (*dataspace_id > 0) {
      H5Sclose( *dataspace_id );
      *dataspace_id = 0;
    }
    if (*dataset_id > 0) {
      H5Dclose( *dataset_id );
      *dataset_id = 0;
    }
  }
  return;
}
