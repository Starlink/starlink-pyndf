/*
*+
*  Name:
*     dat1GetAttrString

*  Purpose:
*     Retrieve string scalar from an attribute

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     dat1GetAttrString( hid_t objid, const char * attrname, hdsbool_t usedef,
*                        const char * defval, char * attrval, size_t attrvallen,
*                        int *status);

*  Arguments:
*     objid = hid_t (Given)
*        HDF5 object containing the attribute.
*     attrname = const char * (Given)
*        Name of attribute.
*     usedef = hdsbool_t (Given)
*        If the attribute is missing, use the default value
*        if this parameter is true.
*     defval = const char * (Given)
*        Default value to use for the attribute. Only used if
*        usedef is true. Copied into output buffer if used.
*     attrval = char * (Given & Returned)
*        String buffer to receive the nul-terminated result.
*     attrvallen = size_t (Given)
*        Allocated length of attrval buffer.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Read a scalar string from the named attribute and return it
*     via the supplied buffer. Use a default value if the attribute is missing.

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
#include "star/one.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

void
dat1GetAttrString( hid_t objid, const char * attrname, hdsbool_t usedef,
                   const char * defval, char * attrval, size_t attrvallen,
                   int *status) {

  hid_t attrtype = 0;
  hid_t attribute_id = 0;
  size_t lenstr = 0;

  if (*status != SAI__OK) return;

  if (!attrval) {
    *status = DAT__FATAL;
    emsRep("dat1GetAttrString_0", "Supplied buffer seems to be a NULL pointer"
           " (possible programming error)", status );
    return;
  }
  attrval[0] = '\0';

  /* We have to get the string type from the attribute and determine its length */
  /* This means we have to get the attribute details so do the existance check
     here */
  if (!H5Aexists(objid, attrname)) {
    if (usedef) {
      one_strlcpy( attrval, defval, attrvallen, status );
    } else {
        *status = DAT__OBJIN;
        emsRepf("dat1GetAttrInt", "Could not retrieve mandatory string attribute from '%s'", status, attrname);
    }
    return;
  }

  /* Get the attribute object */
  CALLHDF( attribute_id,
           H5Aopen( objid, attrname, H5P_DEFAULT ),
           DAT__HDF5E,
           emsRepf("dat1GetAttrString_1", "Error retrieving attribute named %s",
                   status, attrname)
           );
  /* and the data type */
  CALLHDF( attrtype,
           H5Aget_type(attribute_id),
           DAT__HDF5E,
           emsRepf("dat1GetAttrString_2", "Error retrieving data type of attributed name %s",
                   status, attrname)
           );

  /* Check the size [include room for the \0] */
  lenstr = H5Tget_size( attrtype );
  if (lenstr >= attrvallen) {
    if (*status == SAI__OK) {
      *status = DAT__TRUNC;
      emsRepf("dat1GetAttrString_3", "Supplied buffer is too small to receive attribute (%zu must be < %zu)",
              status, lenstr, attrvallen);
    }
    goto CLEANUP;
  }

  /* Read the attribute */
  (void) dat1GetAttr( objid, attrname, attrtype, 1, attrval, NULL,
                      status );

  /* Terminate the string */
  attrval[lenstr] = '\0';

 CLEANUP:
  if (attrtype > 0) H5Tclose(attrtype);
  if (attribute_id > 0) H5Aclose(attribute_id);
  return;
}
