/*
*+
*  Name:
*     dat1GetAttrHdsdims

*  Purpose:
*     Retrieve array of hdsdim types from an attribute

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     dat1GetAttrHdsdims( hid_t objid, const char * attrname, hdsbool_t usedef,
*                         size_t ndefault, const hdsdim defdims[], size_t maxvals,
*                         hdsdim values[], size_t *actvals, int *status);

*  Arguments:
*     objid = hid_t (Given)
*        HDF5 object containing the attribute.
*     attrname = const char * (Given)
*        Name of attribute.
*     usedef = hdsbool_t (Given)
*        If the attribute is missing, use the default value
*        if this parameter is true.
*     ndefault = size_t (Given)
*        Number of elements in the defdims array. Only used if
*        usedef is true.
*     defdims = const hdsdim [] (Given)
*        Default values to use for the attribute. Only used if
*        usedef is true. Can be NULL if usedef is false.
*     maxvals = size_t (Given)
*        Maximum number of values that can be retrieved.
*     values = hdsdim []  (Returned)
*        Buffer to retrieve attribute values. nvals elements (not bytes).
*     actvals = size_t * (Returned)
*        Number of elements retrieved from attribute. Can be NULL.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Read array of hdsdim values  from the named attribute and return it. Use
*     values if the attribute is missing.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Assuming there is not a bug in the HDSv5 library, the main reason
*       the attribute could be missing is if an HDF5 file not created in HDS
*       was given to the HDSv5 library.

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
dat1GetAttrHdsdims( hid_t objid, const char * attrname, hdsbool_t usedef,
                    size_t ndefault, const hdsdim defdims[], size_t maxvals,
                    hdsdim values[], size_t *actvals, int *status) {
  hdsbool_t existed = HDS_FALSE;
  hid_t attrtype = 0;
  hsize_t hvalues[DAT__MXDIM];

  if (*status != SAI__OK) return;

  CALLHDF( attrtype,
           H5Tcopy(H5T_NATIVE_HSIZE),
           DAT__HDF5E,
           emsRepf("dat1GetAttrHdsdims_1", "Error copying data type during reading of attribute '%s'", status, attrname );
           );

  existed = dat1GetAttr( objid, attrname, attrtype, DAT__MXDIM, hvalues, actvals,
                         status );

  if (*status == SAI__OK) {
    if (!existed) {
      if (usedef) {
        if (ndefault > maxvals) {
          *status = DAT__DIMIN;
          emsRepf("dat1GetAttrHdsdims_3", "Missing attribute %s and not enough space to copy %zu default values"
                  " (possible programming error)", status, attrname, ndefault);
        } else {
          size_t i;
          *actvals = ndefault;
          for (i = 0; i < *actvals; i++) {
            values[i] = defdims[i];
          }
        }
      } else {
        *status = DAT__OBJIN;
        emsRepf("dat1GetAttrHdsdims_4", "Could not retrieve mandatory hdsdim attribute array from '%s'", status, attrname);
      }
    } else {
      if (*actvals > maxvals) {
        *status = DAT__DIMIN;
        emsRepf("dat1GetAttrHdsdims_5", "Successfully read %zu values from attribute %s but receiving"
                " array only had space for %zu", status, *actvals, attrname, maxvals);
      } else {
        size_t i;
        for (i = 0; i < *actvals; i++) {
          values[i] = hvalues[i];
        }
      }
    }
  }

 CLEANUP:
  if (attrtype > 0) H5Tclose(attrtype);
  return;
}
