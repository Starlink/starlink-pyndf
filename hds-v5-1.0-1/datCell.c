/*
*+
*  Name:
*     datCell

*  Purpose:
*     Locate cell

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datCell(const HDSLoc *locator1, int ndim, const hdsdim subs[],
*             HDSLoc **locator2, int *status );

*  Arguments:
*     locator1 = const HDSLoc * (Given)
*        Array object locator.
*     ndim = int (Given)
*        Number of dimensions.
*     sub = const hdsdim [] (Given)
*        Subscript values locating the cell in the array. 1-based.
*     locator2 = HDSLoc ** (Returned)
*        Cell locator.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Return a locator to a "cell" (element) of an array object.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     Typically, this is used to locate an element of a structure
*     array for subsequent access to its components, although this
*     does not preclude its use in accessing a single pixel in a 2-D
*     image for example.

*  History:
*     2014-09-06 (TIMJ):
*        Initial version
*     2014-10-28 (TIMJ):
*        Fix case of vectorized 1-D structure.
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
datCell(const HDSLoc *locator1, int ndim, const hdsdim subs[],
        HDSLoc **locator2, int *status ) {

  hsize_t h5subs[DAT__MXDIM];
  HDSLoc * thisloc = NULL;
  int isstruct = 0;
  int rdonly;
  char namestr[DAT__SZNAM+1];
  int lockinfo;

  if (*status != SAI__OK) return *status;

  /* Validate input locator. */
  dat1ValidateLocator( "datCell", 1, locator1, 1, status );

  datName(locator1, namestr, status );

  /* Copy dimensions if appropriate */
  dat1ImportDims( ndim, subs, h5subs, status );

  isstruct = dat1IsStructure( locator1, status );

  /* Validate dimensionality */
  if (*status == SAI__OK) {
    int objndims = 0;
    hdsdim dims[DAT__MXDIM];
    datShape( locator1, DAT__MXDIM, dims, &objndims, status );

    if (objndims == 0) {
      if (*status == SAI__OK) {
        *status = DAT__DIMIN;
        emsRepf("datCell_41", "Can not use datCell for scalar %s '%s' "
                "(possible programming error)", status,
                (isstruct ? "group" : "primitive"), namestr );
      }
    }

    if (objndims != ndim) {
      if (*status == SAI__OK) {
        *status = DAT__DIMIN;
        emsRepf("datCell_1", "datCell: Arguments have %d axes but locator to '%s' refers to %d axes",
                status, ndim, namestr, objndims);
      }
    }
  }

  if (*status != SAI__OK) return *status;


  if (isstruct) {
    char cellname[128];
    hid_t group_id = 0;
    int rank = 0;
    hdsdim groupsub[DAT__MXDIM];

    if (locator1->vectorized > 0) {
      hdsdim structdims[DAT__MXDIM];
      /* If this locator is vectorized then the name will be incorrect
         if we naively calculate the name. */

      /* Get the dimensionality */
      rank = dat1GetStructureDims( locator1, DAT__MXDIM, structdims, status );

      if (rank == 0) {
        /* So the group is really a scalar so we just need to clone the
           input locator */
        datClone( locator1, &thisloc, status );
        goto CLEANUP;
      } else if (rank == 1) {
        /* No special mapping required */
        groupsub[0] = subs[0];

      } else if (rank > 1) {
        /* Map vectorized index to underlying dimensionality */
        dat1Index2Coords( subs[0], rank, structdims, groupsub, status );
        ndim = rank;
      } else {
        if (*status != SAI__OK) {
          *status = DAT__OBJIN;
          emsRepf("datCell_X", "datCell: Rank of structure out of range: %d", status, rank);
        }
        goto CLEANUP;
      }
    } else {
      int i;
      /* Copy the subscripts to a new array so that we can deal with
         the vectorized case above */
      for (i=0; i<ndim; i++) {
        groupsub[i] = subs[i];
      }
    }

    /* Calculate the relevant group name */
    dat1Coords2CellName( ndim, groupsub, cellname, sizeof(cellname), status );
    CALLHDF(group_id,
            H5Gopen2( locator1->group_id, cellname, H5P_DEFAULT ),
            DAT__OBJIN,
            emsRepf("datCell_3", "datCell: Error opening component %s", status, cellname)
            );

    /* Create the locator */
    thisloc = dat1AllocLoc( status );

    if (*status == SAI__OK) {
      /* Handle for cell's data object */
      thisloc->handle = dat1Handle( locator1, cellname, 0, status );

      /* Determine if the current thread has a read-only or read-write lock
         on the parent array, referenced by the supplied locator. */
      dat1HandleLock( locator1->handle, 1, 0, 0, &lockinfo, status );
      rdonly = ( lockinfo == 3 );

      /* Attempt to lock the cell for use by the current thread, using the
         same sort of lock (read-only or read-write) as the parent object.
         Report an error if this fails. */
      dat1HandleLock( thisloc->handle, 2, 0, rdonly, &lockinfo, status );
      if( !lockinfo && *status == SAI__OK ) {
         *status = DAT__THREAD;
         emsSetc( "C", cellname );
         emsSetc( "A", rdonly ? "read-only" : "read-write" );
         datMsg( "O", locator1 );
         emsRep( "","datCell: requested cell ^C within HDS object '^O' "
                 "cannot be locked for ^A access - another thread already has "
                 "a conflicting lock on the same component.", status );
      }

      thisloc->group_id = group_id;
      /* Secondary locator by definition */
      thisloc->file_id = locator1->file_id;
      /* Register it */
      hds1RegLocator( thisloc, status );
    }

  } else {

    /* Just get a slice of one pixel */
    datSlice( locator1, ndim, subs, subs, &thisloc, status );

  }

  /* A single cell is a scalar, and so is not discontiguous. */
  if (*status == SAI__OK) {
    thisloc->iscell = 1;
    thisloc->isdiscont = 0;
  }

 CLEANUP:
  if (*status != SAI__OK) {
    if (thisloc) datAnnul( &thisloc, status );
  } else {
    *locator2 = thisloc;
  }
  return *status;
}
