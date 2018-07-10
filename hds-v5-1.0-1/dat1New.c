/*
*+
*  Name:
*     dat1New

*  Purpose:
*     Create a new component in a structure and return a locator

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     HDSLoc * dat1New( const HDSLoc *locator, hdsbool_t isprimary,
*                       const char *name_str, const char *type_str,
*                       int ndim, const hdsdim dims[], int * status );

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Locator to structure that will receive the new component.
*     isprimary = hdsbool_t (Given)
*        Whether the new locator is primary or secondary locator.
*     name = const char * (Given)
*        Name of the object in the container.
*     type = const char * (Given)
*        Type of object.  If type matches one of the HDS primitive type names a primitive
*        of that type is created, otherwise the object is assumed to be a structure.
*     ndim = int (Given)
*        Number of dimensions. Use 0 for a scalar. See the Notes for a discussion of
*        arrays of structures.
*     dims = const hdsdim [] (Given)
*        Dimensionality of the object. Should be dimensioned with ndim. The array
*        is not accessed if ndim == 0.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Creates a new component (primitive type or structure) in an existing structure
*     and return the corresponding locator.

*  Returned Value:
*     HDSLoc * = locator associated with newly created structure. NULL on error.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Primitive non-scalar types are created with unlimited dimensions to
*       allow resizing but the chunk size is configured to match the supplied
*       dimensions. The HDS API has no means to control whether resizing will
*       be required and how the chunking should be handled.

*  History:
*     2014-08-20 (TIMJ):
*        Initial version
*     2014-09-04 (TIMJ):
*        Unlimited dimensions.
*     2014-09-05 (TIMJ):
*        Add arrays of structures
*     2014-11-14 (TIMJ):
*        Add isprimary argument so that newly created objects can be
*        created as secondary locators.
*     2014-11-22 (TIMJ):
*        Now the HDS root is the HDF5 root group "/"
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

#include <stdlib.h>
#include <strings.h>

#include "hdf5.h"
#include "ems.h"
#include "star/one.h"
#include "prm_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"
#include "dat_err.h"
#include "sae_par.h"

HDSLoc *
dat1New( const HDSLoc    *locator,
         hdsbool_t      isprimary,
        const char      *name_str,
        const char      *type_str,
        int       ndim,
        const hdsdim    dims[],
        int       *status) {

  char cleanname[DAT__SZNAM+1];
  char groupstr[DAT__SZTYP+1];

  hid_t group_id = 0;
  hid_t dataset_id = 0;
  hid_t dataspace_id = 0;
  hid_t cparms = 0;
  hid_t h5type = 0;
  hid_t place = 0;
  int isprim;
  hsize_t h5dims[DAT__MXDIM];

  if (*status != SAI__OK) return NULL;

  /* The name can not have "." in it as this will confuse things
     even though HDF5 will be using a "/" */
  dau1CheckName( name_str, 1, cleanname, sizeof(cleanname), status );
  if (*status != SAI__OK) return NULL;

  /* Copy dimensions if appropriate */
  dat1ImportDims( ndim, dims, h5dims, status );

  /* Work out where to place the component */
  place = dat1RetrieveContainer( locator, status );

  /* Convert the HDS data type to HDF5 data type */
  isprim = dau1CheckType( 0, type_str, &h5type, groupstr,
                          sizeof(groupstr), status );

  /* The above routine has allocated resources so from here we can not
     simply return on error but have to ensure we clean up */

  /* Now create the group or dataset */
  if (isprim) {
    dat1NewPrim( place, ndim, h5dims, h5type, cleanname,
                 &dataset_id, &dataspace_id, status );

    /* If this is intended to be a root locator we indicate this with an attribute
       in the root group to indicate this -- we use an attribute to indicate an HDS
       modification to a default behavior that would be to assume the file root
       is the toplevel group for HDS */
    if (!locator->group_id) {
      dat1SetAttrString( locator->file_id, HDS__ATTR_ROOT_PRIMITIVE, cleanname, status );
    }
  } else {
    /* Create a group -- if the supplied locator is a root locator
     the group already exists and in HDF5 is "/". In that case
     we open the group and write an attribute of the required name. */

    if (!locator->group_id) {
      CALLHDF( group_id,
               H5Gopen2( locator->file_id, "/", H5P_DEFAULT ),
               DAT__HDF5E,
               emsRepf("dat1New_4a", "Error opening root group", status )
               );

      /* Special case -- we need to store the name somewhere */
      dat1SetAttrString( group_id, HDS__ATTR_ROOT_NAME, cleanname, status );

    } else {
      CALLHDF( group_id,
               H5Gcreate2(place, cleanname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
               DAT__HDF5E,
               emsRepf("dat1New_4", "Error creating structure/group '%s'", status, cleanname)
               );
    }

    /* Actual data type of the structure/group must be stored in an attribute */
    dat1SetAttrString( group_id, HDS__ATTR_STRUCT_TYPE, groupstr, status );

    if (ndim > 0) {
      /* HDF5 can not define an array of structures so we create a collection
         of groups below the parent group. */
      int i;
      size_t ngroups = 1;
      size_t n;

      /* Write dimensionality as an attribute */
      dat1SetStructureDims( group_id, ndim, dims, status );

      /* Structures will always be accessed by their coordinates
         (3,2) or (4) or (1,3,2) etc. It makes sense therefore to
         simply name our structures such that these coordinates
         are embedded directly in the name. This has some advantages:
         - We know how to trivially map from the requested coordinate
           to a group.
         - When the name is requested (e.g. hdsTrace) we already
           know that ROOT.RECORDS.HDSCELL(3,2).SOMEINT will have an
           effective trace of ROOT.RECORDS(3,2).SOMEINT [simply remove
           the ".HDSCELL(3,2)" from the full path.
      */

      for (i = 0; i < ndim; i++) {
        ngroups *= h5dims[i];
      }

      for (n = 1; n <= ngroups; n++) {
        hid_t cellgroup_id = 0;
        cellgroup_id = dat1CreateStructureCell( group_id, n, groupstr, cleanname, ndim, dims, status );
        if (cellgroup_id > 0) H5Gclose(cellgroup_id);
      }
    }
  }

  /* We now have to store this in a new locator */
  if (*status == SAI__OK) {
    HDSLoc *thisloc = dat1AllocLoc( status );
    if (*status == SAI__OK) {
      thisloc->handle = dat1Handle( locator, cleanname, 0, status );
      thisloc->dataset_id = dataset_id;
      thisloc->group_id = group_id;
      thisloc->dataspace_id = dataspace_id;
      thisloc->dtype = h5type;
      thisloc->file_id = locator->file_id;
      thisloc->isprimary = isprimary;
      hds1RegLocator( thisloc, status );
    }
    return thisloc;
  }

 CLEANUP:
  /* Everything should be freed */
  if (h5type) H5Tclose( h5type );
  if (dataset_id) H5Dclose(dataset_id);
  if (dataspace_id) H5Sclose(dataspace_id);
  if (cparms > 0 && cparms != H5P_DEFAULT) H5Pclose(cparms);
  if (group_id) H5Gclose(group_id);
  return NULL;
}

