/*
*+
*  Name:
*     hdsOpen

*  Purpose:
*     Open container file

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hdsOpen( const char * file, const char * mode, HDSLoc **locator,
*              int * status );

*  Arguments:
*     file = const char * (Given)
*        Container filename.
*     mode = const char * (Given)
*        Access mode ("READ", "UPDATE" or "WRITE")
*     locator = HDSLoc ** (Returned)
*        Object locator.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Open an existing container file for reading, writing or updating
*     and return a primary locator to the top-level object.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - A file extension of DAT__FLEXT (".h5sdf") is the default.
*     - There is no difference betwee UPDATE and WRITE mode in HDF5
*     - If the top-level object is a structure array, loc will be
*       associated with the complete array, not the first cell. Thus,
*       access to any of the structure's components can only be made
*       through another locator which is explicitly associated with an
*       individual cell (see datCell).
*     - HDS assumes a single dataset/group at the root of the hierarchy.
*       Currently hdsOpen will pick the first valid item if there is a
*       choice. In the future the HDSTYPE attribute might be examined to see
*       which of the top-level items was created by this library.

*  History:
*     2014-08-29 (TIMJ):
*        Initial version
*     2014-11-22 (TIMJ):
*        Root HDF5 group is now the HDS root
*     2018-03-02 (DSB):
*        Allow a file to be opened for read-write access that is already
*        open for read-only access. This uses a starlink-specific feature
*        in HDF5 enabled by supplying the H5F_ACC_FORCERW flag when opening
*        the file.
*     2018-07-03 (DSB):
*        Ensure the returned locator is locked for use by the current thread.
*        Previously, this was not the case if the file was already open in 
*        another thread, in which case the existing Handle was re-used without
*        any further locks being requested.

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

#include <unistd.h>
#include <string.h>

#include "hdf5.h"

#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

int
hdsOpen( const char *file_str, const char *mode_str,
         HDSLoc **locator, int *status) {
  HDSLoc *temploc = NULL;
  Handle *error_handle = NULL;
  Handle *handle = NULL;
  char * fname = NULL;
  hid_t file_id = 0;
  hid_t group_id = 0;
  htri_t filstat = 0;
  unsigned int flags = 0;
  int rdonly = 0;
  int lstat;

  *locator = NULL;
  if (*status != SAI__OK) return *status;

  /* Configure the HDF5 library for our needs as this routine could be called
     before any others. */
  dat1InitHDF5();

  /* Work out the flags for opening. Note if we are opening the file for
     update or write access,  mimic the old HDS behaviour by including
     the starlink-specific H5F_ACC_FORCERW flag. This flag prevents an
     error being reported within H5Fopen if the file has previously been
     opened for read-only access. HDS V4 allows a file to opened for
     update even if it has previously been opened read-only, although
     the second open may fail if the file is write-protected. Some
     starlink apps rely on this behaviour. */
  switch (mode_str[0]) {
  case 'U':
  case 'u':
  case 'w':
  case 'W':
    flags = H5F_ACC_RDWR | H5F_ACC_FORCERW;
    break;
  case 'R':
  case 'r':
    flags = H5F_ACC_RDONLY;
    rdonly = 1;
    break;
  default:
    *status = DAT__MODIN;
  }

  /* work out the file name */
  fname = dau1CheckFileName( file_str, status );

  /* Before we go any further, check that the file really is an HDF5
     file. If it isn't then we return with a special error code that
     allows the putative wrapper library to know to fall back to HDSv4
     or whatever. */
  if (*status != SAI__OK) goto CLEANUP;
  filstat = H5Fis_hdf5( fname );
  if (filstat < 0) {
    /* Probably indicates the file is not there */
    *status = DAT__FILNF;
    emsRepf("hdsOpen_fnf", "File '%s' does not seem to exist",
           status, fname);
    goto CLEANUP;
  }

  /* Open the HDF5 file */
  CALLHDF( file_id,
           H5Fopen( fname, flags, H5P_DEFAULT ),
           DAT__HDF5E,
           emsRepf("hdsOpen_1", "Error opening HDS file: %s",
                   status, fname )
           );

  /* Now we need to find a top-level object. This will usually simply
     be the root group but for the special case where we have an HDS
     primitive in the root group we have to open that one level down. */
  CALLHDF( group_id,
           H5Gopen2(file_id, "/", H5P_DEFAULT),
           DAT__HDF5E,
           emsRepf("hdsOpen_2","Error opening root group of file %s",
                  status, fname)
           );

  /* If the attribute indicating we have to use a primitive as the top
     level is present we open that for the root locator */
  if (H5Aexists( group_id, HDS__ATTR_ROOT_PRIMITIVE)) {
    char primname[DAT__SZNAM+1];
    dat1GetAttrString( group_id, HDS__ATTR_ROOT_PRIMITIVE, HDS_FALSE,
                       NULL, primname, sizeof(primname), status );

    /* Now that we have a name we can use datFind once we have a locator,
       we create a temporary locator but must register it so that an error
       in datFind will not close the file immediately before we close it ourselves later. */
    temploc = dat1AllocLoc( status );
    if (*status == SAI__OK) {
      temploc->file_id = file_id;
      file_id = 0; /* now owned by the locator system */
      temploc->isprimary = HDS_TRUE;
      temploc->group_id = group_id;
      hds1RegLocator( temploc, status );
      handle = hds1FindHandle( temploc->file_id, status );
    }
    datFind( temploc, primname, locator, status );

    /* force the locator to be primary as we get rid of the parent locator */
    if (*status == SAI__OK) (*locator)->isprimary = HDS_TRUE;

  } else {
    /* Turn the root group into a locator */
    temploc = dat1AllocLoc( status );
    if (*status == SAI__OK) {
      temploc->group_id = group_id;
      temploc->file_id = file_id;
      group_id = file_id = 0; /* now owned by the locator system */
      temploc->isprimary = HDS_TRUE;
      hds1RegLocator( temploc, status );
      handle = hds1FindHandle( temploc->file_id, status );
    }
    if (*status == SAI__OK) {
      /* Assign this locator to the caller */
      *locator = temploc;
      temploc = NULL;
    }
  }

  /* If the file was already open via another locator, store the
     top-level handle associated with that locator. Otherwise,
     create a new handle structure for the top level data object in the
     file. Store the handle pointer in the locator. Then attempt to lock
     the handle appropriately (read-only or read-write) for use by the
     current thread (in fact we only need to do this if we are re-using
     a pre-existing handle, since dat1Handle will already have ensured
     any newly created Handle is locked appropriately). */
  if( *locator ) {
    if( !handle ) {
       handle = dat1Handle( NULL, fname, rdonly, status );
    } else {
      error_handle = dat1HandleLock( handle, 2, 0, rdonly, &lstat, status );
      if( error_handle && *status == SAI__OK ) {
         *status = DAT__THREAD;
         emsSetc( "U", rdonly ? "read-only" : "read-write" );
         emsSetc( "O", file_str );
         emsRep( " ", "hdsOpen: Cannot lock HDS object '^O' for ^U use by "
                 "the current thread:", status );
         dat1HandleMsg( "E", error_handle );
         if( error_handle != handle ) {
            emsRep( " ", "A component within it (^E) is locked for writing by another thread.", status );
         } else {
            emsRep( " ", "It is locked for writing by another thread.", status );
         }
      }
    }
    (*locator)->handle = handle;
  }

 CLEANUP:
  if (fname) MEM_FREE(fname);

  /* Free the temporary which will close the parent group */
  if (temploc) datAnnul(&temploc, status );

  if (*status != SAI__OK) {
    /* cleanup */
    if (*locator) {
       (*locator)->handle = dat1EraseHandle( (*locator)->handle, NULL, status );
        datAnnul( locator, status );
    }

    if (file_id > 0) H5Fclose( file_id );
  }

  return *status;

}
