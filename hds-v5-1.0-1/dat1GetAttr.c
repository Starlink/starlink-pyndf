/*
*+
*  Name:
*     dat1GetAttr

*  Purpose:
*     Retrieve values of specified data type from an HDF5 attribute

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     exists = dat1GetAttr( hid_t obj_id, const char * attrname, hid_t attrtype,
*                           size_t maxvals, void * values, size_t *actvals, int * status );

*  Arguments:
*     obj_id = hid_t (Given)
*        HDF5 object to associate with attribute.
*     attrname = const char * (Given)
*        Name of attribute.
*     attrtype = hid_t (Given)
*        Memory data type of attribute.
*     maxvals = size_t (Given)
*        Maximum number of values that can be retrieved.
*     values = void *  (Returned)
*        Buffer to retrieve attribute values. maxvals elements (not bytes).
*     actvals = size_t * (Returned)
*        Number of elements retrieved from attribute. Can be NULL.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Retrieve values of a specified data type from the HDF5 attribute.
*     An error will be triggered if the size of the attribute dataspace
*     exceeds maxvals.

*  Returned Value:
*     Returns 1 if an attribute was found, 0 if an attribute was
*     not found. The absence of an attribute is not an error and it is
*     up to the caller to decide how to react.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-11-17 (TIMJ):
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
dat1GetAttr( hid_t obj_id, const char * attrname, hid_t attrtype,
             size_t maxvals, void * values, size_t *actvals, int * status ) {

  hid_t attribute_id = 0;
  hid_t attr_dataspace_id = 0;

  if (*status != SAI__OK) return HDS_FALSE;

  /* Check for existance and return immediately if not there */
  if (!H5Aexists(obj_id, attrname)) return HDS_FALSE;

  if (!values) {
    *status = DAT__FATAL;
    emsRepf("dat1GetAttr", "Can not retrieve attribute '%s' into a null pointer"
            " (possible programming error)", status, attrname);
    abort();
    return HDS_TRUE;
  }

  /* Get the attribute object */
  CALLHDF( attribute_id,
           H5Aopen( obj_id, attrname, H5P_DEFAULT ),
           DAT__HDF5E,
           emsRepf("dat1GetAttr_1", "Error retrieving attribute named %s",
                   status, attrname)
           );

  /* Retrieve the underlying dataspace */
  CALLHDF( attr_dataspace_id,
           H5Aget_space( attribute_id ),
           DAT__HDF5E,
           emsRepf("dat1GetAttr_2", "Error retrieving dataspace from attribute named %s",
                   status, attrname)
           );

  {
    size_t nelem = 1;
    hsize_t dims[DAT__MXDIM];
    int rank;
    rank = H5Sget_simple_extent_ndims( attr_dataspace_id );
    if (rank > DAT__MXDIM) {
      if (*status == SAI__OK) {
        *status = DAT__DIMIN;
        emsRepf("dat1GetAttr_3", "Can not have more than %d dimensions in an HDS attribute, got %d",
                status, DAT__MXDIM, rank );
      }
      goto CLEANUP;
    }
    if (rank > 0) { /* not a scalar */
      int i;
      CALLHDF(rank,
              H5Sget_simple_extent_dims( attr_dataspace_id, dims, NULL),
              DAT__HDF5E,
              emsRepf("dat1GetAttr_4", "Error retrieving dimensions of attribute %s", status, attrname)
              );
      for (i=0; i<rank; i++) {
        nelem *= dims[i];
      }
    }
    if (nelem > maxvals) {
      *status = DAT__DIMIN;
      emsRepf("dat1GetAttr_4", "Supplied buffer to small to retrieve %zu values from attribute %s",
              status, nelem, attrname);
      goto CLEANUP;
    }
    if (actvals) *actvals = nelem;
  }

  /* Now copy out the data */
  CALLHDFQ(H5Aread( attribute_id, attrtype, values ));

 CLEANUP:
  if (attribute_id > 0) H5Aclose(attribute_id);
  if (attr_dataspace_id > 0) H5Sclose(attr_dataspace_id);

  return HDS_TRUE;
}
