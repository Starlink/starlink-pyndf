/*
*+
*  Name:
*     datAlter

*  Purpose:
*     Alter object size

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datAlter( HDSLoc *locator, int ndim, const hdsdim dims[], int *status);

*  Arguments:
*     locator = HDSLoc * (Given)
*        Object locator to alter.
*     ndim = int (Given)
*        Number of dimensions specified in "dim". Must be the number of dimensions
*        in the object itself.
*     dim = const hdsdim [] (Given)
*        New dimensions for object.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Alter the size of an array by increasing or reducing the last
*     (or only) dimension. If a structure array is to be reduced in
*     size, the operation will fail if any truncated elements contain
*     components.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - The dimensionality of the object can not differ before and after this
*     routine is called.
*     - Can not be called on a vectorized locator.

*  History:
*     2014-10-14 (TIMJ):
*        Initial version
*     2014-10-29 (TIMJ):
*        Now can reshape structure arrays.
*     2014-11-05 (TIMJ):
*        Resize by creating a new dataset and copying across
*        and deleting the original. Required for memory mapping.
*     2014-11-06 (TIMJ):
*        Try the native resize first. It might actually work and
*        it will be more efficient. It will only work if the dataset
*        could not support memory mapping so the fact that the new
*        one also won't is irrelevant.
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
#include "star/one.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

int
datAlter( HDSLoc *locator, int ndim, const hdsdim dims[], int *status) {

  hid_t h5type = 0;
  hdsdim curdims[DAT__MXDIM];
  int curndim;
  int i;
  HDSLoc * parloc = NULL;
  HDSLoc * temploc = NULL;
  hid_t new_dataset_id = 0;
  hid_t new_dataspace_id = 0;
  int rdonly;
  int lockinfo;

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datAlter", 1, locator, 0, status );

  if (locator->vectorized) {
    *status = DAT__OBJIN;
    emsRep("datAlter_1", "Can not alter the size of a vectorized object",
           status);
    return *status;
  }

  if (locator->regpntr) {
    *status = DAT__OBJIN;
    emsRep("datAlter_2", "Can not alter the size of a mapped primitive",
           status);
    return *status;
  }

  if (locator->isslice) {
    *status = DAT__OBJIN;
    emsRep("datAlter_3", "Can not alter the size of a slice",
           status);
    return *status;
  }

  /* Get the current dimensions and validate new ones */
  datShape( locator, DAT__MXDIM, curdims, &curndim, status );

  if (curndim != ndim) {
    if (*status == SAI__OK) {
      *status = DAT__DIMIN;
      emsRepf("datAlter_4", "datAlter can not change the dimensionality (%d != %d)",
              status, curndim, ndim);
      return *status;
    }
  }

  if (*status != SAI__OK) return *status;

  for (i=0; i< (ndim-2); i++) {
    if ( dims[i] != curdims[i] ) {
      *status = DAT__DIMIN;
      emsRepf("datAlter_5", "datAlter: Dimension %d (1-based) does not match "
              "(%" HDS_DIM_FORMAT " != %" HDS_DIM_FORMAT ")",
              status, i+1, dims[i], curdims[i]);
      return *status;
    }
  }

  if ( dat1IsStructure( locator, status) ) {
    hdsdim curcount = 0;
    hdsdim newcount = 0;

    /* The restriction on final dimension being altered simplifies
       the calculation somewhat in that we can work in vectorized
       space. If we have 10 elements now and need to have 8 then
       we know we just delete elements 9 and 10. If we require
       12 we know we just add 11 and 12 at the correct coordinates. */
    curcount = 1;
    newcount = 1;
    for (i=0; i<curndim; i++) {
      curcount *= curdims[i];
      newcount *= dims[i];
    }

    if (newcount > curcount) {
      /* Need to extend */
      char grouptype[DAT__SZTYP+1];
      char groupname[DAT__SZNAM+1];
      datType( locator, grouptype, status );
      datName( locator, groupname, status );

      for (i=curcount+1; i <= newcount; i++) {
        hid_t cellgroup_id = 0;
        cellgroup_id = dat1CreateStructureCell( locator->group_id, i, grouptype, groupname, ndim, dims, status );
        if (cellgroup_id > 0) H5Gclose(cellgroup_id);
      }
    } else if (newcount < curcount) {
      /* Need to shrink - delete each structure and complain if
         the structure is not empty -- use curdims */
      for (i=newcount+1; i<=curcount; i++) {
        hdsdim coords[DAT__MXDIM];
        char cellname[128];
        HDSLoc * cell = NULL;
        int ncomp = 0;
        dat1Index2Coords(i, ndim, curdims, coords, status );
        dat1Coords2CellName( ndim, coords, cellname, sizeof(cellname), status );

        /* Need to peak inside -- datCell would be a bit inefficient but use minimum code/
           Should still work as I remove earlier structures as part of the loop */
        datCell(locator, ndim, coords, &cell, status );
        datNcomp( cell, &ncomp, status );
        datAnnul( &cell, status );
        if (ncomp > 0) {
          if (*status == SAI__OK) {
            *status = DAT__DELIN;
            emsRep("datAlter_6", "datAlter: Can not shrink structure array as some structures"
                   " to be deleted contain components", status );
          }
          goto CLEANUP;
        }
        /* Remove the empty element -- can not use datErase because
           structure elements are deliberately too long. */
        CALLHDFQ( H5Ldelete( locator->group_id, cellname, H5P_DEFAULT ) );
      }
    } else {
      /* Oddly, no change requested so we are done. Should this be an error? */
      goto CLEANUP;
    }

    /* Need to update the dimensions in the attribute */
    dat1SetStructureDims( locator->group_id, ndim, dims, status );

  } else {
    hsize_t h5dims[DAT__MXDIM];
    char primname[DAT__SZNAM+1];
    char tempname[3*DAT__SZNAM+1];
    hdsbool_t state;
    herr_t h5err;

    /* Copy dimensions and reorder */
    dat1ImportDims( ndim, dims, h5dims, status );

    /* First we simply try the native resize. This will only work
       if the system is using chunked storage and the registered
       upper limit to the bounds is acceptable. If it fails we will
       just fall back to the long-winded inefficient version. */
    h5err = H5Dset_extent( locator->dataset_id, h5dims );
    if (h5err >= 0) {
      /* Actually worked so we need to define a new dataspace */
      H5Sclose( locator->dataspace_id );
      locator->dataspace_id = H5Dget_space( locator->dataset_id );
    } else {
      /* The native resize failed. This is the most likely scenario
         for HDS when we have configured the system to attempt to
         create datasets that can be memory mapped. We therefore
         resize by creating a new dataset of the correct size,
         copying in the contents from the original, deleting the
         original, then renaming the new dataset. */

      /* Need enclosing group locator */
      datParen( locator, &parloc, status );

      /* HDF5 data type of the locator */
      CALLHDF( h5type,
               H5Dget_type( locator->dataset_id ),
               DAT__HDF5E,
               emsRep("dat1Type_1", "datType: Error obtaining data type of dataset", status)
               );

      /* Create a new dataset with a name related to this dataset but which
         does not conform to the HDS rules */
      datName( locator, primname, status );
      one_snprintf(tempname, sizeof(tempname), "%s%s", status,
                   "+TEMPORARY_DATASET_", primname);

      /* If the entry already exists this *must* be due to a previous
         error condition in datAlter so we throw up our hands and assume we
         can delete it */
      if (H5Lexists( parloc->group_id, tempname, H5P_DEFAULT)) {
        H5Ldelete( parloc->group_id, tempname, H5P_DEFAULT);
      }

      dat1NewPrim( parloc->group_id, ndim, h5dims, h5type, tempname,
                   &new_dataset_id, &new_dataspace_id, status );

      /* Nothing to copy if the source locator is not defined */
      datState( locator, &state, status );
      if (state && *status == SAI__OK) {
        size_t numin = 1;
        size_t numout = 1;
        size_t nbperel;
        char type_str[DAT__SZTYP+1];
        void *inpntr = NULL;
        void *outpntr = NULL;
        size_t nbytes = 0;

        /* Type in HDS speak */
        datType(locator, type_str, status );

        /* Number of bytes per element -- should be the same in and out */
        datLen( locator, &nbperel, status );

        /* Easiest to map the the input and output and copy */
        datMapV( locator, type_str, "READ", &inpntr, &numin, status );

        /* Create temporary locator for output -- do not need to register this
           and we are not annulling it. */
        temploc = dat1AllocLoc( status );
        temploc->dataset_id = new_dataset_id;
        temploc->dataspace_id = new_dataspace_id;
        temploc->file_id = locator->file_id;
        temploc->handle = locator->handle;

        /* And map the output */
        datMapV( temploc, type_str, "WRITE", &outpntr, &numout, status );

        /* Check we can use outpntr safely. */
        if( *status == SAI__OK ) {

           /* Copy up to numin elements */
           nbytes = nbperel * (numin > numout ? numout : numin);
           memcpy( outpntr, inpntr, nbytes );

           /* Then set the remaining elements to 0 (or bad). We could
              get around this need by specifying the fill value on object creation */
           if ( numout > numin) {
             size_t nextra = nbperel * (numout - numin);
             unsigned char * offpntr = NULL;
             offpntr = &((unsigned char *)outpntr)[nbytes];
             memset( offpntr, 0, nextra );
           }
        }

        /* Unmap and free the temporary locator */
        datUnmap( locator, status );
        datUnmap( temploc, status );
        temploc = dat1FreeLoc( temploc, status );
      }

      /* Determine if the current thread has a read-only or read-write lock
         on the supplied object referenced by the supplied locator. */
      dat1HandleLock( locator->handle, 1, 0, 0, &lockinfo, status );
      rdonly = ( lockinfo == 3 );

      /* Delete the source dataset -- free resources in supplied locator */
      H5Sclose( locator->dataspace_id );
      H5Dclose( locator->dataset_id );
      datErase( parloc, primname, status );

      /* Relocate the new dataset */
      CALLHDFQ(H5Lmove( parloc->group_id, tempname,
                        parloc->group_id, primname, H5P_DEFAULT, H5P_DEFAULT));

      /* Update the locator */
      locator->dataspace_id = new_dataspace_id;
      locator->dataset_id = new_dataset_id;

      /* Give it a new handle (the old one will have been modified or
         erased within datErase above). */
      locator->handle = dat1Handle( parloc, primname, rdonly, status );

      /* Attempt to lock the locator again for use by the current thread,
         using the same sort of lock (read-only or read-write) as the
         supplied locator. Report an error if this fails. */

      dat1HandleLock( locator->handle, 2, 0, rdonly, &lockinfo, status );
      if( !lockinfo && *status == SAI__OK ) {
         *status = DAT__THREAD;
         emsSetc( "A", rdonly ? "read-only" : "read-write" );
         emsRep( " ","datAlter: altered object cannot be locked for ^A "
                 "access.", status );
      }

    }
  }

 CLEANUP:
  datAnnul(&parloc, status);
  if (*status != SAI__OK) {
    if (h5type > 0) H5Tclose( h5type );
    if (temploc) temploc = dat1FreeLoc( temploc, status );
  }
  return *status;
}
