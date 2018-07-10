/* Wrapper source file that contains both the routine for registering
 * a locator with a group (hdsLink) and the corresponding routine for
 * freeing locators in a group. The two are combined to allow the routines
 * to easily share a single data structure for group membership.
 */

#include <pthread.h>
#include "ems.h"
#include "sae_par.h"
#include "star/one.h"
#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

/* Have a simple hash table keyed by group name with
 * values being an array of locators. HDS groups are not
 * used very often (mainly in ADAM) so for now there is no
 * need for high performance data structures. We do store
 * the group name in the locator struct to allow fast
 * retrieval of the group name without having to scan
 * all the hash tables.
 */

/* Use the uthash macros: https://github.com/troydhanson/uthash */
#include "uthash.h"
#include "utarray.h"

/* We do not want to clone so we just copy the pointer */
/* UTarray takes a copy so we create a mini struct to store
   the actual pointer. */
typedef struct {
  HDSLoc * locator;    /* Actual HDS locator */
} HDSelement;
static UT_icd locators_icd = { sizeof(HDSelement *), NULL, NULL, NULL };

typedef struct {
  char grpname[DAT__SZGRP+1]; /* Group name: the key */
  UT_array * locators;        /* Pointer to hash of element structs */
  UT_hash_handle hh;          /* Mandatory for UTHASH */
} HDSgroup;

/* Declare the hash */
static HDSgroup *groups = NULL;

/* Prototypes for private functions */
static int hds2Link( HDSLoc *locator, const char *group_str, int *status );
static int hds2Flush( const char *group_str, int *status );
static hdsbool_t hds2RemoveLocator( const HDSLoc * loc, int *status );



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
*     hdsLink

*  Purpose:
*     Link locator group

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hdsLink(HDSLoc *locator, const char *group_str, int *status);

*  Arguments:
*     locator = HDSLoc * (Given and Returned)
*        Object locator
*     group = const char * (Given)
*        Group name.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Link a locator to a group.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - See also hdsFlush and hdsGroup.
*     - Once a locator is registered with a group it should not be annuled
*       by the caller. It can only be annuled by calling hdsFlush.
*     - A locator can only be assigned to a single group.

*  History:
*     2014-10-17 (TIMJ):
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


int hdsLink(HDSLoc *locator, const char *group_str, int *status) {
   LOCK_MUTEX;
   int result = hds2Link( locator, group_str, status );
   UNLOCK_MUTEX
   return result;
}



/*
*+
*  Name:
*     hdsFlush

*  Purpose:
*     Flush locator group

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hdsFlush( const char *group_str, int *status);

*  Arguments:
*     group = const char * (Given)
*        Group name
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Annuls all locators currently assigned to a specified locator group.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - See also hdsLink and hdsGroup

*  History:
*     2014-10-17 (TIMJ):
*        Initial version
*     2014-11-07 (TIMJ):
*        Remove from group before calling datAnnul
*     2018-07-03 (DSB):
*        Do nothing if the old and new groups names are equal. 
*        This means no read-write lock needs to be acquired, which 
*        can prevent errors from happening (e.g. if the locator is 
*        already locked by another thread).
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

int hdsFlush( const char *group_str, int *status) {
   LOCK_MUTEX;
   int result = hds2Flush( group_str, status );
   UNLOCK_MUTEX
   return result;
}


/* Remove a locator from a group. This is currently a private
   routine to allow datAnnul to free a locator that has been
   associated with a group outside of hdsFlush. This is quite
   probably a bug but a bug that is currently prevalent in
   SUBPAR which seems to store locators in groups and then
   frees them anyhow. This removal will prevent hdsFlush
   attempting to also free the locator.

   Returns true if the locator was removed. Do nothing if the locator is
   not part of a group.

*/

hdsbool_t hds1RemoveLocator( const HDSLoc * loc, int *status ) {
   hdsbool_t result = 1;
   if ( (loc->grpname)[0] != '\0') {
      LOCK_MUTEX;
      result = hds2RemoveLocator( loc, status );
      UNLOCK_MUTEX
   }
   return result;
}











/* Private functions
   -----------------------------------------------------------------------
   These can only be called from the public functions called above and so
   do not need to be serialised. The API for each function is the same as
   that documented above for the corresponding public function. */

static int hds2Link(HDSLoc *locator, const char *group_str, int *status) {
  HDSgroup *entry = NULL;
  HDSelement elt;
  memset(&elt, 0, sizeof(elt));
  int lock_status;
  int ok;
  int promoted;
  int locked;

  if (*status != SAI__OK) return *status;

  /* If we get a zero length string this either means we are trying to unlink
     the locator from the group or we have a bug in the calling code or else
     we mean that we don't want to link the locator at all. For now trigger
     an error */
  if (!group_str || strlen(group_str) == 0) {
    *status = DAT__GRPIN;
    emsRep("hdsLink_2", "Supplied group name is empty or null", status );
    return *status;
  }

  /* Validate supplied locator, but do not check that the current thread
     is locked by the current thread. */
  dat1ValidateLocator( "hdsLink", 0, locator, 0, status );

  /* If the requested group name is already set, return without further action.
     This avoids us needing a read-write lock on the locator and so avoids
     unnecessary failures in cases where the locator is already locked by a
     different thread. */
  if( !locator->grpname || strcmp( locator->grpname, group_str ) ) {

     /* We only need to check the locator's lock if the group name is
        currently set and so will be changed by this call. This is safe
        (i.e. two threads cannot set a name simultaneously) because this
        function is already serialised by a mutex. */
     promoted = 0;
     locked = 0;
     if ( (locator->grpname)[0] != '\0') {

       /* The group name is stored inside the locator, so changing the group modifies
          the locator structure. Therefore, we need to ensure that the current
          thread has a read/write lock on the locator. If it does, continue.
          If it does not, but it has the only read lock on the locator, we
          temporarily promote the read lock to a read/write lock. First get
          the lock status of the locator. */
       dat1HandleLock( locator->handle, 1, 0, 0, &lock_status, status );

       /* If it is locked by the current thread for reading but not writing, we
          attempt to promote the read lock to a read/write lock. Report an error if
          the promotion fails (i.e. because another thread also has a read-lock). */
       ok = 0;
       if( lock_status == 3 ){
          dat1HandleLock( locator->handle, 2, 0, 0, &lock_status, status );
          if( lock_status == 1 ) {
             ok = 1;
             promoted = 1;
          }

       /* If it is locked by the current thread for reading and writing, we
          can continue withotu changing anything. */
       } else if( lock_status == 1 ){
          ok = 1;

       /* If it is unlocked we temporarily lock it for reading and writing by
          the current thread. */
       } else if( lock_status == 0 ){
          dat1HandleLock( locator->handle, 2, 0, 0, &lock_status, status );
          if( lock_status == 1 ) {
             ok = 1;
             locked = 1;
          }
       }

       /* If we cannot get a write-lock report an error and return. */
       if( !ok && *status == SAI__OK ){
          *status = DAT__THREAD;
          datMsg( "O", locator );
          emsRepf( " ", "hdsLink: The supplied HDS locator for '^O' cannot be used.",
                   status );
          emsRep( " ", "It cannot be locked for read-write access by the current "
                  "thread (programming error).", status );
          return *status;
       }

     /* We are moving  a locator to a different group so unregister it
        from the old group first. */
       hds2RemoveLocator(locator, status);
     }

     /* Now copy the group name to the locator */
     one_strlcpy( locator->grpname, group_str, sizeof(locator->grpname), status );

     /* See if this entry already exists in the hash */
     HASH_FIND_STR( groups, group_str, entry );
     if (!entry) {
       entry = calloc( 1, sizeof(HDSgroup) );
       one_strlcpy( entry->grpname, group_str, sizeof(entry->grpname), status );
       utarray_new( entry->locators, &locators_icd);
       HASH_ADD_STR( groups, grpname, entry );
     }

     /* Now we have the entry, we need to store the locator inside.
        We do not clone the locator, the locator is now owned by the group. */
     elt.locator = locator;
     utarray_push_back( entry->locators, &elt );

     /* If the locator was originally unlocked, unlock it now. */
     if( locked ){
        dat1HandleLock( locator->handle, 3, 0, 1, &lock_status, status );
        if( lock_status != 1 && *status == SAI__OK ) {
           *status = DAT__THREAD;
           datMsg( "O", locator );
           emsRepf( " ", "hdsLink: The supplied HDS locator for '^O' cannot be used.",
                    status );
           emsRep( " ", "The read-write lock cannot be unlocked (programming error).", status );
        }
     }

     /* If the locator was originally promoted from a read lock to a
        read/write lock, demote it back to a read lock. */
     if( promoted ){
        dat1HandleLock( locator->handle, 2, 0, 1, &lock_status, status );
        if( lock_status != 1 && *status == SAI__OK ) {
           *status = DAT__THREAD;
           datMsg( "O", locator );
           emsRepf( " ", "hdsLink: The supplied HDS locator for '^O' cannot be used.",
                    status );
           emsRep( " ", "The read-write lock cannot be demoted to a "
                   "read-only lock(programming error).", status );
        }
     }
  }

  return *status;
}





static int hds2Flush( const char *group_str, int *status) {
  HDSgroup * entry = NULL;
  HDSelement * elt = NULL;

  if (*status != SAI__OK) return *status;

  /* See if this entry already exists in the hash */
  HASH_FIND_STR( groups, group_str, entry );
  if (!entry) {
    *status = DAT__GRPIN;
    emsRepf("hdsFlush_1", "Can not flush a group named '%s' that does not exist",
            status, group_str );
    return *status;
  }

  /* Read all the elements from the entry and annul them */
  for ( elt = (HDSelement *)utarray_front(entry->locators);
        elt != NULL;
        elt = (HDSelement *)utarray_next(entry->locators, elt )) {
    HDSLoc * loc = elt->locator;
    /* clear the group membership -- otherwise datAnnul will
       call back to the group code to try to remove the locator */
    (loc->grpname)[0] = '\0';
    datAnnul( &loc, status );
  }

  /* Free the array and delete the hash entry */
  utarray_free( entry->locators );
  HASH_DEL( groups, entry );

  return *status;
}





static hdsbool_t hds2RemoveLocator( const HDSLoc * loc, int *status ) {
  HDSgroup * entry = NULL;
  HDSelement * elt = NULL;
  const char * grpname;
  int pos = -1;
  unsigned int len = 0;
  unsigned int i = 0;
  int removed = 0;

  if (*status != SAI__OK) return removed;

  /* Not associated with a group */
  grpname = loc->grpname;
  if (grpname[0] == '\0') return removed;

  /* Look for the entry associated with this name */
  HASH_FIND_STR( groups, grpname, entry );
  if ( !entry ) return removed;

  len = utarray_len( entry->locators );
  /* Read all the elements from the entry, looking for the relevant one */
  for ( i = 0; i < len; i++) {
    HDSLoc * thisloc;
    elt = (HDSelement *)utarray_eltptr( entry->locators, i );
    thisloc = elt->locator;
    if (thisloc == loc) {
      pos = i;
      break;
    }
  }

  if (pos > -1) {
    unsigned int upos = pos;
    utarray_erase( entry->locators, upos, 1 );
    removed = 1;
  }

  return removed;
}




