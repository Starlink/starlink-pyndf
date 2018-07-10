/*
*+
*  Name:
*     dat1SetAttr

*  Purpose:
*     Store values of specified data type in an HDF5 attribute

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     dat1SetAttr( hid_t obj_id, const char * attrname, hid_t attrtype,
*                  size_t nvals, const void * values, int * status );

*  Arguments:
*     obj_id = hid_t (Given)
*        HDF5 object to associate with attribute.
*     attrname = const char * (Given)
*        Name of attribute.
*     attrtype = hid_t (Given)
*        Data type of attribute.
*     nvals = size_t (Given)
*        Number of values to store. If 0 a scalar dataspace is defined
*        and one value is assumed.
*     values = const void * (Given)
*        Value to store in attribute.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Store values in an attribute associated
*     with the specified HDF5 object using the specified data type.

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

void
dat1SetAttr( hid_t obj_id, const char * attrname, hid_t attrtype,
             size_t nvals, const void * values, int * status ) {

  hid_t attribute_id = 0;
  hid_t attr_dataspace_id = 0;

  if (*status != SAI__OK) return;

  if (!values) {
    *status = DAT__FATAL;
    emsRepf("dat1SetAttr", "Can not set attribute '%s' with a null pointer"
            " (possible programming error)", status, attrname);
    return;
  }

  if (nvals == 0) {
    CALLHDF(attr_dataspace_id,
            H5Screate( H5S_SCALAR ),
            DAT__HDF5E,
            emsRepf("dat1SetAttrString_2", "Error creating data space for attribute '%s'", status, attrname )
            );
  } else {
    hsize_t h5dims[1];
    h5dims[0] = nvals;
    CALLHDF( attr_dataspace_id,
             H5Screate_simple( 1, h5dims, NULL ),
             DAT__HDF5E,
             emsRepf("dat1New_1", "Error allocating data space for attribute %s",
                     status, attrname )
             );
  }

  if (H5Aexists( obj_id, attrname)) H5Adelete( obj_id, attrname );

  CALLHDF(attribute_id,
          H5Acreate2( obj_id, attrname, attrtype, attr_dataspace_id,
                      H5P_DEFAULT, H5P_DEFAULT),
          DAT__HDF5E,
          emsRepf("dat1SetAttrString_3", "Error creating attribute named '%s'", status, attrname );
          );

  CALLHDFQ(H5Awrite( attribute_id, attrtype, values ));

 CLEANUP:
  if (attribute_id > 0) H5Aclose(attribute_id);
  if (attr_dataspace_id > 0) H5Sclose(attr_dataspace_id);

  return;
}
