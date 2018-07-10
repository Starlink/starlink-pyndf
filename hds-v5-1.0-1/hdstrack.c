/* Single source file to provide facilities for registering a locator
 * and unregistering a locator. This allows us to keep track of
 * all active locators and, more importantly, allows us to track
 * primary/secondary locator status: once a file id no longer has
 * any primary locators associated with it the file will be closed. */

/* Single source file as that simplifies the sharing of the uthash.
   Implementation similar to hdsgroups but here we use the HDF5 file_id
   as the key. */

#include <string.h>
#include <pthread.h>

#include "hdf5.h"
#include "ems.h"
#include "sae_par.h"
#include "star/one.h"
#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

/* Have a simple hash table keyed by file_id with
 * values being an array of locators.
 */

/* Use the uthash macros: https://github.com/troydhanson/uthash */
#include "uthash.h"
#include "utarray.h"

/* Because we are using a non-standard data type as hash
   key we have to define new macros */
#define HASH_FIND_FILE_ID(head,findfile,out)    \
  HASH_FIND(hh,head,findfile,sizeof(hid_t),out)
#define HASH_ADD_FILE_ID(head,filefield,add)    \
  HASH_ADD(hh,head,filefield,sizeof(hid_t),add)

/* We do not want to clone so we just copy the pointer */
/* UTarray takes a copy so we create a mini struct to store
   the actual pointer. */
typedef struct {
  HDSLoc * locator;    /* Actual HDS locator */
} HDSelement;

static UT_icd all_locators_icd = { sizeof(HDSelement *), NULL, NULL, NULL };

typedef struct {
  hid_t file_id;              /* File id: the key */
  UT_array * locators;        /* Pointer to hash of element structs */
  UT_hash_handle hh;          /* Mandatory for UTHASH */
} HDSregistry;

/* Declare the hash */
static HDSregistry *all_locators = NULL;

/* Private routines */
static Handle *hds2FindHandle( hid_t file_id, int *status );
static Handle *hds2TopHandle( Handle *handle );
static hdsbool_t hds2UnregLocator( HDSLoc * locator, int *status );
static hid_t *hds2GetFileIds( hid_t file_id, int *status );
static int hds2CountFiles();
static int hds2CountLocators( size_t ncomp, char **comps, hdsbool_t skip_scratch_root, int * status );
static int hds2FlushFile( hid_t file_id, int *status );
static int hds2FlushFileID( hid_t file_id, int *status);
static int hds2GetFileDescriptor( hid_t file_id );
static int hds2RegLocator(HDSLoc *locator, int *status);
static size_t hds2PrimaryCount( hid_t file_id, int *status );
static size_t hds2PrimaryCountByFileID( hid_t file_id, int * status );
static void hds2ShowFiles( hdsbool_t listfiles, hdsbool_t listlocs, int *status );
static void hds2ShowLocators( hid_t file_id, int * status );




/* Public functions
   -----------------------------------------------------------------------
   These use a mutex to serialise calls so that the module variables used
   within this module are not accessed by multiple threads at the same
   time. There is a one-to-one correspondance between these public
   functions and the corresponding private function. */

static pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
#define LOCK_MUTEX pthread_mutex_lock( &mutex1 );
#define UNLOCK_MUTEX pthread_mutex_unlock( &mutex1 );


/*
*+
*  Name:
*     hds1RegLocator

*  Purpose:
*     Register locator

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hds1RegLocator(const HDSLoc *locator, int *status);

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Object locator
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Register a new locator.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - See also hds2UnregLocator.

*  History:
*     2014-11-10 (TIMJ):
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

int hds1RegLocator(HDSLoc *locator, int *status) {
   LOCK_MUTEX;
   int result = hds2RegLocator( locator, status );
   UNLOCK_MUTEX
   return result;
}

/*
*+
*  Name:
*     hds1FlushFile

*  Purpose:
*     Annul all locators associated with a specific open file

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hds1FlushFile( hid_t file_id, int *status);

*  Arguments:
*     file_id = hid_t (Given)
*        File identifier. Used to determine all file identifiers
*        associated with the file.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Annuls all locators currently assigned to a specified file.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - See also hds1RegLocator
*     - Will annul all locators and will not attempt
*       to check that there is more than one primary locator.

*  History:
*     2014-11-11 (TIMJ):
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

int hds1FlushFile( hid_t file_id, int *status ) {
   LOCK_MUTEX;
   int result = hds1FlushFile( file_id, status );
   UNLOCK_MUTEX
   return result;
}

/* Remove a locator from the master list.
   Helper routine for datAnnul which can remove the locator
   from the master list. If this was the last locator associated
   with the file or if it is the last primary locator associated
   with the file, the file will be closed and all other locators
   associated with the file will be annulled.

   Returns true if the locator was removed.

*/
hdsbool_t hds1UnregLocator( HDSLoc *locator, int *status ) {
   LOCK_MUTEX;
   hdsbool_t result = hds2UnregLocator( locator, status );
   UNLOCK_MUTEX
   return result;
}

/* Count how many primary locators are associated with a particular
   file -- since a file can have multiple file_ids this routine
   goes through them all. */
size_t hds1PrimaryCount( hid_t file_id, int *status ) {
   LOCK_MUTEX;
   size_t result = hds2PrimaryCount( file_id, status );
   UNLOCK_MUTEX
   return result;
}


/* Version of hdsShow that uses the internal list of locators
   rather than the HDF5 list of locators. This should duplicate
   the HDF5 list. */

void hds1ShowFiles( hdsbool_t listfiles, hdsbool_t listlocs, int * status ) {
   LOCK_MUTEX;
   hds2ShowFiles( listfiles, listlocs, status );
   UNLOCK_MUTEX
}

void hds1ShowLocators( hid_t file_id, int * status ) {
   LOCK_MUTEX;
   hds2ShowLocators( file_id, status );
   UNLOCK_MUTEX
}

int hds1CountFiles() {
   LOCK_MUTEX;
   int result = hds2CountFiles();
   UNLOCK_MUTEX
   return result;
}



/*
  ncomp = number of components in search filter
  comps = char ** - array of pointers to filter strings
  skip_scratch_root = true, skip HDS_SCRATCH root locators
*/

int hds1CountLocators( size_t ncomp, char **comps, hdsbool_t skip_scratch_root,
                       int * status ) {
   LOCK_MUTEX;
   int result = hds2CountLocators( ncomp, comps, skip_scratch_root, status );
   UNLOCK_MUTEX
   return result;
}

/*
*+
*  Name:
*     hds1FindHandle

*  Purpose:
*     Find a Handle for the top-level object in a specified file.

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     Handle *hds1FindHandle( hid_t file_id, int *status )

*  Arguments:
*     file_id = hid_t (Given)
*        HDF5 file identifier.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Returned function value:
*     Pointer to the existing Handle that describes the top level data
*     object in the file specified by "file_id", or NULL if the file has
*     not previously been opened and so has no top-level Handle as yet.
*     Also returns NULL if an error occurs.

*  Description:
*     A Handle is a structure that contains ancillary information about
*     each physical data object that is shared by all locators that
*     refer to the object. Each locator contains a pointer to the Handle
*     that describes the associated physical data object. In other words,
*     if a change is made to the values in a Handle, then that change is
*     immediately visible through all locators that refer to that data
*     object. By comparison, a change made to the values in a single
*     locator are not visible to other locators that refer to the same
*     object.
*
*     Since files can be opened more than once in HDS and HDF5, we need
*     to make sure that the same Handle structure is used each time a
*     particular file is opened. This function facilitiates this by
*     searching for an existing locator that refers to the file specified
*     by argument "file_id". If the file has not previously been opened,
*     no such locator will be found and NULL is returned as the function
*     value. If the file has already been opened, a locator will be found
*     and will contain a pointer to a Handle for the corresponding
*     physical data object. Handle structures contain links that link
*     them all together into a tree structure. If a locator is found,
*     these links are navigated to find the top of the tree and a pointer
*     to the associated Handle is returned as the function value.
*
*     This function should be called after opening a new HDS file, but
*     before the top-level locator for that file is registered using
*     hds1RegLocator.

*  Authors:
*     DSB: David S Berry (EAO)
*     {enter_new_authors_here}

*  History:
*     6-JUL-2017 (DSB):
*        Initial version
*     {enter_further_changes_here}

*  Copyright:
*     Copyright (C) 2017 East Asian Observatory
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

Handle *hds1FindHandle( hid_t file_id, int *status ){
   LOCK_MUTEX;
   Handle *result = hds2FindHandle( file_id, status );
   UNLOCK_MUTEX
   return result;
}





/* Private functions
   -----------------------------------------------------------------------
   These can only be called from the public functions called above and so
   do not need to be serialised. The API for each function is the same as
   that documented above for the corresponding public function. */

static int hds2RegLocator(HDSLoc *locator, int *status) {
  HDSregistry *entry = NULL;
  HDSelement elt;
  hid_t file_id = 0;
  memset(&elt, 0, sizeof(elt));

  if (*status != SAI__OK) return *status;

  /* Sanity check */
  if (locator->file_id <= 0) {
    *status = DAT__FATAL;
    emsRep("hds2RegLocator_1", "Can not register a locator that is not"
           " associated with a file", status );
    return *status;
  }

  /* See if this entry already exists in the hash */
  file_id = locator->file_id;
  HASH_FIND_FILE_ID( all_locators, &file_id, entry );
  if (!entry) {
    entry = calloc( 1, sizeof(HDSregistry) );
    entry->file_id = locator->file_id;
    utarray_new( entry->locators, &all_locators_icd);
    HASH_ADD_FILE_ID( all_locators, file_id, entry );
  }

  /* Now we have the entry, we need to store the locator inside.
     We do not clone the locator, the locator is now owned by the group. */
  elt.locator = locator;
  utarray_push_back( entry->locators, &elt );
  return *status;
}

static int hds2FlushFile( hid_t file_id, int *status ) {
  hid_t *file_ids = 0;
  size_t i = 0;
  if (*status != SAI__OK) return *status;

  /* Get all the file ids associated with this individual file_id */
  file_ids = hds2GetFileIds( file_id, status );

  /* Get count from each file_id */
  while (file_ids[i]) {
    hds2FlushFileID( file_ids[i], status );
    i++;
  }

  if (file_ids) MEM_FREE(file_ids);
  return *status;
}


static int hds2FlushFileID( hid_t file_id, int *status) {
  HDSregistry * entry = NULL;
  HDSelement * elt = NULL;

  if (*status != SAI__OK) return *status;

  /* See if this entry already exists in the hash */
  HASH_FIND_FILE_ID( all_locators, &file_id, entry );
  if (!entry) {
    *status = DAT__GRPIN;
    emsRepf("hdsFlush_1", "Can not flush a file %lld that does not exist",
            status, (long long)file_id );
    return *status;
  }

  /* Read all the elements from the entry and annul them */
  for ( elt = (HDSelement *)utarray_front(entry->locators);
        elt != NULL;
        elt = (HDSelement *)utarray_next(entry->locators, elt )) {
    HDSLoc * loc = elt->locator;
    /* clear the file information in the locator as we do not
       want to confuse datAnnul when it then attempts to unregister
       this file from the active list. */
    loc->file_id = 0;

    /* Auto-annull the locator. We use dat1Annul so as not to free the
       C struct memory itself. This results in a memory leak but
       protects against the case where a primary locator is freed
       whilst the user still assumes they have a secondary locator
       to annul. We probably need a new HDS routine to allow
       these resources to be freed at the end of a monolith action
       (similar to an ndfBegin/ndfEnd pairing). To do that we will
       need to store the auto-annulled pointer somewhere. */
    dat1Annul( loc, status );
  }

  /* Now we close the file itself */
  H5Fclose(file_id);

  /* Free the array and delete the hash entry */
  utarray_free( entry->locators );
  HASH_DEL( all_locators, entry );

  return *status;
}

/* Remove a locator from the master list.
   Helper routine for datAnnul which can remove the locator
   from the master list. If this was the last locator associated
   with the file or if it is the last primary locator associated
   with the file, the file will be closed and all other locators
   associated with the file will be annulled.

   Returns true if the locator was removed.

*/

static hdsbool_t hds2UnregLocator( HDSLoc * locator, int *status ) {
  HDSregistry * entry = NULL;
  HDSelement * elt = NULL;
  hid_t file_id = -1;
  int pos = -1;
  unsigned int len = 0;
  unsigned int i = 0;
  int removed = 0;

  if (*status != SAI__OK) return removed;

  /* Not associated with a file, should not happen */
  if (locator->file_id <= 0) {
    *status = DAT__FATAL;
    emsRep("hds2RegLocator_1", "Can not unregister a locator that is not"
           " associated with a file", status );
    return *status;
  }

  /* Look for the entry associated with this name */
  file_id = locator->file_id;
  HASH_FIND_FILE_ID( all_locators, &file_id, entry );
  if ( !entry ) {
    /* if we did not find it for now trigger an error because
       this code is meant to have a store of all locators that
       were allocated */
    if (*status == SAI__OK) {
      *status = DAT__FATAL;
      emsRepf("hds2UnregLocator_2", "Internal error with locator tracking"
              " (Possible programming error)", status );
    }
    return removed;
  }

  len = utarray_len( entry->locators );
  /* Read all the elements from the entry, looking for the relevant one */
  /* Do not count primary/secondary as that needs to be done across
     multiple file handles */
  for ( i = 0; i < len; i++) {
    HDSLoc * thisloc;
    elt = (HDSelement *)utarray_eltptr( entry->locators, i );
    thisloc = elt->locator;
    if (thisloc == locator) {
      /* Can break straight away */
      pos = i;
      break;
    }
  }

  if (pos > -1) {
    size_t nprimary;
    unsigned int upos = pos;

    /* Remove it from the list so that there will not be a later
       attempt to remove it.*/
    utarray_erase( entry->locators, upos, 1 );

    /* This locator is being annulled so it's okay to remove
       it from the list. We remove the file_id element so as not
       to confuse datAnnul into thinking it should close the file. */
    locator->file_id = 0;

    /* Get the TOTAL count for this file (not just this file_id)
       (without this locator, which we just removed) */
    nprimary = hds2PrimaryCount( file_id, status );

    /* Trigger a cleanup if there are no more primary locators
       -- we call flush even if we know this is the
       only locator so that we do not duplicate the hash delete code */
    if (nprimary == 0) {

      /* Get the handle at the top of the tree and erase the whole tree. */
      dat1EraseHandle( hds2TopHandle( locator->handle ), NULL, status );

      /* Close all locators */
      hds2FlushFile( file_id, status );
    }

    removed = 1;
  } else {
    /* Somehow we did not find the locator. This should not happen. */
    if (*status == SAI__OK) {
      *status = DAT__WEIRD;
      emsRep("hds2UnregLocator", "Could not find locator associated with file"
             " (possible programming error)", status);
    }
  }

  return removed;
}

/* Count how many primary locators are associated with a particular
   file -- since a file can have multiple file_ids this routine
   goes through them all. */
static size_t hds2PrimaryCount( hid_t file_id, int *status ) {
  size_t nprimary = 0;
  hid_t *file_ids = NULL;
  size_t i = 0;
  if (*status != SAI__OK) return nprimary;

  /* Get all the file ids associated with this individual file_id */
  file_ids = hds2GetFileIds( file_id, status );

  /* Get count from each file_id */
  while (file_ids[i]) {
    nprimary += hds2PrimaryCountByFileID(file_ids[i], status);
    i++;
  }

  if (file_ids) MEM_FREE(file_ids);
  return nprimary;
}



/* Count how many primary locators are associated with a particular
   file_id */

static size_t hds2PrimaryCountByFileID( hid_t file_id, int * status ) {
  HDSregistry * entry = NULL;
  HDSelement * elt = NULL;
  unsigned int len = 0;
  unsigned int i = 0;
  size_t nprimary = 0;

  if (*status != SAI__OK) return nprimary;

  /* Look for the entry associated with this name */
  HASH_FIND_FILE_ID( all_locators, &file_id, entry );

  /* Possibly should be an error */
  if ( !entry ) return nprimary;

  len = utarray_len( entry->locators );
  /* Read all the elements from the entry, looking for the relevant one
     but also counting how many primary locators we have. */
  for ( i = 0; i < len; i++) {
    HDSLoc * thisloc;
    elt = (HDSelement *)utarray_eltptr( entry->locators, i );
    thisloc = elt->locator;
    if (thisloc->isprimary) nprimary++;
  }

  return nprimary;
}


/* Version of hdsShow that uses the internal list of locators
   rather than the HDF5 list of locators. This should duplicate
   the HDF5 list.
*/

static void hds2ShowFiles( hdsbool_t listfiles, hdsbool_t listlocs, int * status ) {
  HDSregistry *entry;
  unsigned int num_files;
  if (*status != SAI__OK) return;

  num_files = HASH_COUNT(all_locators);
  printf("Internal HDS registry: %u file%s\n", num_files, (num_files == 1 ? "" : "s"));
  for (entry = all_locators; entry != NULL; entry = entry->hh.next) {
    unsigned intent = 0;
    hid_t file_id = 0;
    unsigned int len = 0;
    char * name_str = NULL;
    const char * intent_str = NULL;
    size_t nprim = 0;
    file_id = entry->file_id;
    H5Fget_intent( file_id, &intent );
    if (intent == H5F_ACC_RDONLY) {
      intent_str = "R";
    } else if (intent == H5F_ACC_RDWR) {
      intent_str = "U";
    } else {
      intent_str = "Err";
    }
    len = utarray_len( entry->locators );
    nprim = hds2PrimaryCountByFileID( file_id, status );
    name_str = dat1GetFullName( file_id, 1, NULL, status );
    if (listfiles) printf("File: %s [%s] (%d) (%u locator%s) (refcnt=%zu)\n", name_str, intent_str, file_id,
                          len, (len == 1 ? "" : "s"), nprim);
    if (listlocs) hds2ShowLocators( file_id, status );
    if (name_str) MEM_FREE(name_str);
  }
}

static void hds2ShowLocators( hid_t file_id, int * status ) {
  HDSregistry * entry = NULL;
  HDSelement * elt = NULL;
  unsigned int len = 0;
  unsigned int i = 0;
  size_t nprimary = 0;

  if (*status != SAI__OK) return;

  /* Look for the entry associated with this name */
  HASH_FIND_FILE_ID( all_locators, &file_id, entry );

  /* Possibly should be an error */
  if ( !entry ) return;

  len = utarray_len( entry->locators );
  /* Read all the elements from the entry, looking for the relevant one
     but also counting how many primary locators we have. */
  printf("File %d has %u locator%s:\n", file_id, len, (len == 1 ? "" : "s"));
  for ( i = 0; i < len; i++) {
    HDSLoc * thisloc;
    char * namestr = NULL;
    hid_t objid = 0;
    elt = (HDSelement *)utarray_eltptr( entry->locators, i );
    thisloc = elt->locator;
    objid = dat1RetrieveIdentifier( thisloc, status );
    if (objid > 0) namestr = dat1GetFullName( objid, 0, NULL, status );
    printf("Locator %p [%s] (%s) group=%s\n", thisloc, (namestr ? namestr : "no groups/datasets"),
           (thisloc->isprimary ? "primary" : "secondary"),thisloc->grpname);
    if (thisloc->isprimary) nprimary++;
    MEM_FREE(namestr);
  }
}

/* Retrieve the file descriptor of the underlying file */
static int hds2GetFileDescriptor( hid_t file_id ) {
  int fd = 0;
  hid_t fapl_id;
  hid_t fdriv_id;
  void *file_handle;
  herr_t herr = -1;
  fapl_id = H5Fget_access_plist(file_id);
  fdriv_id = H5Pget_driver(fapl_id);
  herr = H5Fget_vfd_handle( file_id, fapl_id, (void**)&file_handle);
  if (herr >= 0) {
    if (fdriv_id == H5FD_SEC2) {
      fd = *((int *)file_handle);
    } else if (fdriv_id == H5FD_STDIO) {
      FILE * fh = (FILE *)file_handle;
      fd = fileno(fh);
    }
  }
  if (fapl_id > 0) H5Pclose(fapl_id);
  return fd;
}

/* This routine looks up all the file_ids associated with
   the same file as the supplied file id. Currently uses
   the underlying file descriptor for matching. This should
   be more efficient and more accurate than getting the file name
   and matching that. Might have issues if non-standard file drivers
   are used later on. */

static hid_t *hds2GetFileIds( hid_t file_id, int *status ) {
  hid_t * file_ids = NULL;
  size_t num_files;
  size_t nfound = 0;
  HDSregistry * entry = NULL;
  int ref_fd = 0;

  if (*status != SAI__OK) return NULL;

  /* Inefficient: Allocate enough memory to hold all open file_ids plus
     one so we can end with a NULL. */
  num_files = hds2CountFiles();
  file_ids = MEM_CALLOC( num_files + 1, sizeof(*file_ids) );
  if (!file_ids) {
    *status = DAT__NOMEM;
    emsRep("GetFileIds", "Serious issue allocating array of file identifiers",
           status );
    goto CLEANUP;
  }

  /* Get this filename as the reference -- assume normalized */
  ref_fd = hds2GetFileDescriptor( file_id );
  file_ids[0] = file_id;
  nfound++;
  if (ref_fd == 0) {
     if( *status == SAI__OK ) {
        *status = DAT__FATAL;
        emsRepf( " ", "hds2GetFileIds: Unexpected null file reference for "
                 "ID %d (internal HDS programming error)", status, file_id );
     }

  } else {
     int file_id_found = 0;

     for (entry = all_locators; entry != NULL; entry = entry->hh.next) {
       hid_t this_file_id = entry->file_id;
       int fd;

       /* We know this matches */
       if (this_file_id == file_id) {
          file_id_found = 1;
          continue;
       }

       fd = hds2GetFileDescriptor( this_file_id );
       if (fd == ref_fd) {
         file_ids[nfound] = this_file_id;
         nfound++;
       }
     }

     if( !file_id_found && *status == SAI__OK ) {
        *status = DAT__FATAL;
        emsRepf( " ", "hds2GetFileIds: File ID %d not found - has the "
                 "corresponding locator been registered? (internal HDS "
                 "programming error).", status, file_id );
     }
  }

 CLEANUP:
  return file_ids;
}

static int hds2CountFiles() {
  int num_files;
  num_files = HASH_COUNT(all_locators);
  return num_files;
}

/*
  ncomp = number of components in search filter
  comps = char ** - array of pointers to filter strings
  skip_scratch_root = true, skip HDS_SCRATCH root locators
*/

static int hds2CountLocators( size_t ncomp, char **comps,
                              hdsbool_t skip_scratch_root, int * status ) {

  HDSregistry * entry = NULL;
  int nlocator = 0;
  if (*status != SAI__OK) return nlocator;

  for (entry = all_locators; entry != NULL; entry = entry->hh.next) {
    unsigned int len = 0;
    hid_t file_id = entry->file_id;
    /* Look for the entry associated with this name */
    HASH_FIND_FILE_ID( all_locators, &file_id, entry );
    if (!entry) continue;
    len = utarray_len( entry->locators );

    /* if we do not have to filter the list then we just add that to the sum */
    if (ncomp == 0) {
      nlocator += len;
    } else {
      unsigned int i;
      /* Loop over each locator in the utarray */
      for ( i = 0; i < len; i++) {
        HDSLoc * thisloc;
        char path_str[1024];
        char file_str[1024];
        HDSelement * elt = NULL;
        int nlev;
        elt = (HDSelement *)utarray_eltptr( entry->locators, i );
        thisloc = elt->locator;

        /* filter provided so we have to check the path of each locator against
           the filter */
        hdsTrace(thisloc, &nlev, path_str, file_str, status, sizeof(path_str),
                 sizeof(file_str));

        if (*status == SAI__OK) {
          /* Now we compare the trace to the filters */
          hdsbool_t match = 0;
          hdsbool_t exclude = 0;

          if (skip_scratch_root) {
            const char *root = "HDS_SCRATCH.TEMP_";
            const size_t rootlen = strlen(root);
            if (strncmp( path_str, root, rootlen) == 0)  {
              /* exclude if the string only has one "." */
              if ( !strstr( &((path_str)[rootlen-1]), ".")) {
                exclude = 1;
              }
            } else if (strcmp( path_str, "HDS_SCRATCH") == 0) {
              /* HDS seems to hide the underlying root locator of the
                 temp file (the global primary locator) and does not
                 even report it with hdsShow -- we skip it here */
              exclude = 1;
            }
          }

          if (!exclude) {
            size_t j;
            for (j=0; j<ncomp; j++) {
              /* matching or anti-matching? */
              if ( *(comps[j]) == '!' ) {
                /* do not forget to start one character in for the ! */
                if (strncmp(path_str, (comps[j])+1,
                            strlen(comps[j])-1) == 0) {
                  /* Should be exempt */
                  exclude = 1;
                }
              } else {
                if (strncmp(path_str, comps[j], strlen(comps[j])) == 0) {
                  /* Should be included */
                  match = 1;
                }
              }
            }
          }

          /* increment if we either matched something
             or was not excluded */
          if (match || !exclude ) nlocator++;

        } else {
          /* plough on regardless */
          emsAnnul(status);
        }
      }
    }
  }

  return nlocator;
}



/*
*+
*  Name:
*     hds1FindHandle

*  Purpose:
*     Find a Handle for the top-level object in a specified file.

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     Handle *hds1FindHandle( hid_t file_id, int *status )

*  Arguments:
*     file_id = hid_t (Given)
*        HDF5 file identifier.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Returned function value:
*     Pointer to the existing Handle that describes the top level data
*     object in the file specified by "file_id", or NULL if the file has
*     not previously been opened and so has no top-level Handle as yet.
*     Also returns NULL if an error occurs.

*  Description:
*     A Handle is a structure that contains ancillary information about
*     each physical data object that is shared by all locators that
*     refer to the object. Each locator contains a pointer to the Handle
*     that describes the associated physical data object. In other words,
*     if a change is made to the values in a Handle, then that change is
*     immediately visible through all locators that refer to that data
*     object. By comparison, a change made to the values in a single
*     locator are not visible to other locators that refer to the same
*     object.
*
*     Since files can be opened more than once in HDS and HDF5, we need
*     to make sure that the same Handle structure is used each time a
*     particular file is opened. This function facilitiates this by
*     searching for an existing locator that refers to the file specified
*     by argument "file_id". If the file has not previously been opened,
*     no such locator will be found and NULL is returned as the function
*     value. If the file has already been opened, a locator will be found
*     and will contain a pointer to a Handle for the corresponding
*     physical data object. Handle structures contain links that link
*     them all together into a tree structure. If a locator is found,
*     these links are navigated to find the top of the tree and a pointer
*     to the associated Handle is returned as the function value.
*
*     This function should be called after opening a new HDS file, but
*     before the top-level locator for that file is registered using
*     hds1RegLocator.

*  Authors:
*     DSB: David S Berry (EAO)
*     {enter_new_authors_here}

*  History:
*     6-JUL-2017 (DSB):
*        Initial version
*     {enter_further_changes_here}

*  Copyright:
*     Copyright (C) 2017 East Asian Observatory
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

static Handle *hds2FindHandle( hid_t file_id, int *status ){

/* Local Variables: */
   HDSelement *elt = NULL;
   HDSregistry *entry = NULL;
   Handle *result = NULL;
   hid_t *file_ids = NULL;
   int i;

/* Check inherited status */
   if( *status != SAI__OK ) return result;

/* Get all the file ids that are associated with the same file as the
   supplied file_id */
   file_ids = hds2GetFileIds( file_id, status );

/* All file ids for the same file should share the same Handle, so we
   only need to look at the first file id - if any. */
   i = 0;
   while( *status == SAI__OK && file_ids[ i ] ) {

/* Find the entry containing the locators that refer to the i'th file
   id. */
      HASH_FIND_FILE_ID( all_locators, file_ids + i, entry );

/* Get a locator - any locator - that refers to the file, and get it's
   Handle. */
      if( entry ) {
         elt = (HDSelement *) utarray_front( entry->locators );

/* Work up the tree of handles to find the top level Handle. */
         if( elt && elt->locator ) result = hds2TopHandle( elt->locator->handle );

/* Leave the file id loop now if we have a Handle. */
         if( result ) break;
      }

/* If we do not yet have a locator (e.g. because the file id does not yet
   have any active locators), look at the next file id. */
      i++;
   }

/* Free the array of file ids. */
   if( file_ids ) MEM_FREE( file_ids );

/* Return the top level Handle pointer. */
   return result;
}

/* Get the handle at the top of the tree containing a specified handle. */
static Handle *hds2TopHandle( Handle *handle ) {
   Handle *parent = NULL;
   Handle *result = handle;

   if( !handle ) return result;

   parent = result->parent;
   while( parent ) {
      result = parent;
      parent = result->parent;
   }

   return result;
}

