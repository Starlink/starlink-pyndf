/*
*+
*  Name:
*     hdsNew

*  Purpose:
*     Create new container file

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     int hdsNew( const char *file_str, const char * name_str, const char * type_str,
*                 int ndim, const hdsdim dims[], HDSLoc **locator, int * status );

*  Arguments:
*     file = const char * (Given)
*        Container file name. Use DAT__FLEXT (".h5sdf") if no suffix specified.
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
*     locator = HDSLoc ** (Returned)
*        HDS locator of the root element.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Creates a new HDS container file and returns a locator to the root element.
*     The file is opened with read/write access and will overwrite any previously
*     existing file.

*  Returned Value:
*     int = inherited status on exit. This is for compatibility with the original
*           HDS API.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - A file extension of DAT__FLEXT (".h5sdf") is the default.
*     - HDF5 file opened with mode H5F_ACC_TRUNC.
*     - HDF5 does not know how to create arrays of structures. When the HDS layer is asked
*     to create a structure array a group (in the HDF5 sense) is created of that name
*     with the string "_STRUCTURE_ARRAY" appended. Inside this group further groups are
*     created named "NAME_1", "NAME_2". Multi-dimensional structure arrays are not
*     supported at this time until a need can be demonstrated. The "locator" returned will
*     be that of the "_STRUCTURE_ARRAY" group and datCell must be used to locate a sub-element.

*  History:
*     2014-08-15 (TIMJ):
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

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "hds1.h"
#include "dat1.h"
#include "ems.h"
#include "dat_err.h"
#include "hds.h"
#include "sae_par.h"

#include "star/one.h"

#include "hdf5.h"

int
hdsNew(const char *file_str,
       const char *name_str,
       const char *type_str,
       int  ndim,
       const hdsdim dims[],
       HDSLoc **locator,
       int *status) {

  char cleanname[DAT__SZNAM+1];
  char groupstr[DAT__SZTYP+1];
  hid_t file_id = 0;
  hsize_t h5dims[DAT__MXDIM];
  HDSLoc * thisloc = NULL;
  hid_t h5type = 0;
  char *fname = NULL;

  /* Returns the inherited status for compatibility reasons */
  if (*status != SAI__OK) return *status;

  /* Configure the HDF5 library for our needs as this routine could be called
     before any others. */
  dat1InitHDF5();

  /* The name can not have "." in it as this will confuse things
     even though HDF5 will be using a "/" */
  dau1CheckName( name_str, 1, cleanname, sizeof(cleanname), status );
  if (*status != SAI__OK) return *status;

  /* Copy dimensions if appropriate */
  dat1ImportDims( ndim, dims, h5dims, status );

  /* Convert the HDS data type to HDF5 data type as an early sanity
     check. */
  (void) dau1CheckType( 0, type_str, &h5type, groupstr,
                        sizeof(groupstr), status );
  if (h5type) H5Tclose( h5type ); /* we are not using this type for real */

  /* The above routine has allocated resources so from here we can not
     simply return on error but have to ensure we clean up */

  /* Create buffer for file name so that we include the file extension */
  fname = dau1CheckFileName( file_str, status );

  /* Create the HDF5 file */
  CALLHDF( file_id,
           H5Fcreate( fname, H5F_ACC_TRUNC,
                      H5P_DEFAULT, H5P_DEFAULT ),
           DAT__FILCR,
           emsRepf("hdsNew","Error creating file '%s'", status, fname )
           );

  /* Create the top-level structure/primitive */
  if (*status == SAI__OK) {
    HDSLoc *tmploc = dat1AllocLoc( status );
    if (*status == SAI__OK) {
      tmploc->file_id = file_id;
      tmploc->isprimary = HDS_TRUE;
      hds1RegLocator( tmploc, status );
      if (*status == SAI__OK) file_id = 0; /* handed file to locator */

      /* Create a new Handle structure describing the new object and store
         it in the locator. Lock it for read-write access by the current
         thread. */
      tmploc->handle = dat1Handle( NULL, fname, 0, status );

      /* We use dat1New instead of datNew so that we do not have to follow
         up immediately with a datFind */
      thisloc = dat1New( tmploc, 1, name_str, type_str, ndim, dims, status );

      /* Annul the temporary locator. The file will not close if
         we still have a primary from the dat1New */
      datAnnul( &tmploc, status );
    }
  }

  /* Return the locator */
  if (*status == SAI__OK) {
    *locator = thisloc;
    return *status;
  }

 CLEANUP:
  /* Free allocated resource */
  /* This includes attempting to delete the new file */
  if (thisloc) {
     thisloc->handle = dat1EraseHandle( thisloc->handle, NULL, status );
     datAnnul( &thisloc, status );
  }
  if (*status != SAI__OK) unlink(fname);
  if (file_id > 0) H5Fclose(file_id);
  if (fname) MEM_FREE(fname);

  return *status;
}


