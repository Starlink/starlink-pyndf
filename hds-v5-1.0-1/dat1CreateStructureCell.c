/*
*+
*  Name:
*     dat1CreateStructureCell

*  Purpose:
*     Create a single element of a structure array

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     dat1CreateStructureCell( hid_t group_id, size_t index, const char * typestr, const char * parentstr,
*                              int ndim, const hdsdim dims[], int *status );

*  Arguments:
*     group_id = hid_t (Given)
*        HDF5 group identifier of the parent structure that will receive the elements.
*     index = size_t (Given)
*        Index into the vectorized array of structures. Given the dimensional information
*        this index is converted to a coordinate that is used to name the group.
*     typestr = const char * (Given)
*        HDS type to associate with the structure. Should match the parent.
*     parentstr = const char * (Given)
*        Name of the parent structure. Used for error messages.
*     ndim = int (Given)
*        Number of dimensions in the dims[] array.
*     dims = const hdsdim [] (Given)
*        Size of each dimension.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Arrays of structures are implemented as individual HDF5 groups that
*     are named by their coordinates within the structure. This routine
*     determines the name from an index position and creates it with the
*     correct data type.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-10-29 (TIMJ):
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

#include "dat_err.h"

hid_t
dat1CreateStructureCell( hid_t group_id, size_t index, const char * typestr, const char * parentstr,
                         int ndim, const hdsdim dims[], int *status ) {

  hid_t cellgroup_id = 0;
  char cellname[128];
  hdsdim coords[DAT__MXDIM];

  if (*status != SAI__OK) return cellgroup_id;

  /* Note we have to use the HDS dims (Fortran order) order for naming */
  dat1Index2Coords(index, ndim, dims, coords, status );
  dat1Coords2CellName( ndim, coords, cellname, sizeof(cellname), status );

  CALLHDF( cellgroup_id,
           H5Gcreate2(group_id, cellname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
           DAT__HDF5E,
           emsRepf("dat1New_4", "Error creating structure/group '%s'", status, parentstr)
           );

  /* Actual data type of the structure/group must be stored in an attribute.
     Do not need to store dimensions as each cell is itself scalar. */
  dat1SetAttrString( cellgroup_id, HDS__ATTR_STRUCT_TYPE, typestr, status );

 CLEANUP:
  if (*status != SAI__OK) {
    if (cellgroup_id > 0) {
      H5Gclose(cellgroup_id);
      cellgroup_id = 0;
    }
  }
  return cellgroup_id;
}
