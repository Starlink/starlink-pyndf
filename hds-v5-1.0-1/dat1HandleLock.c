/*
*+
*  Name:
*     dat1HandleLock

*  Purpose:
*     Manage the lock on a Handle.

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     Handle *dat1HandleLock( Handle *handle, int oper, int recurs,
*                             int rdonly, int *result, int *status );

*  Arguments:
*     handle = Handle * (Given)
*        Handle to be checked.
*     oper = int (Given)
*        Operation to be performed:
*
*        1 - Returns "result" holding information about the current locks
*            on the supplied Handle.
*
*            0: unlocked;
*            1: locked for writing by the current thread;
*            2: locked for writing by another thread;
*            3: locked for reading by the current thread (other threads
*               may also have a read lock on the Handle);
*            4: locked for reading by one or more other threads (the
*               current thread does not have a read lock on the Handle);
*
*            If "recurs" is non-zero and there are child Handles the
*            above values take on the following meanings:
*
*            0: The supplied Handle and all children are unlocked;
*            1: The supplied Handle and all children are locked for writing
*               by the current thread;
*            2: The supplied Handle or one of its children is locked for
*               writing by another thread;
*            3: The supplied Handle and all children are locked for reading
*               by the current thread (other threads may also have a read
*               lock on one or more of the Handles);
*            4: The supplied Handle and all children are locked for reading
*               by one or more other threads (maybe different threads).
*            5: Some mix not covered by the above list.
*
*        2 - Lock the handle (and all descendants if "recurs" is non-zero)
*            for read-write or read-only use by the current thread. The
*            result is 0 if the request conflicts with any existing lock (in
*            which case the request to lock the supplied Handle is ignored)
*            and +1 otherwise.
*        3 - Unlock the handle (and all descendants if "recurs" is
*            non-zero). If the current thread has a lock - either read-write
*            or read-only - on the Handle, it is removed and +1 is returned
*            as the result. Otherwise the Handle is left unchanged and either
*            0 or -1 is returned - -1 is returned if the handle is currently
*            locked for writing by a different thread, and zero is returned
*            otherwise (in which case the current thread does not have a lock,
*            but some other threads may have read locks). If "recurs" is
*            non-zero, +1 is returned only if the supplied Handle and all
*            child handles are locked by the current thread. Otherwise, the
*            returned value relates to the first handle (either the supplied
*            or a child handle) that was no locked by the current thread.
*
*     recurs = int (Given)
*        If "recurs" is zero, the supplied Handle is the only Handle to be
*        checked or modified - any child Handles are ignored. If "recurs" is
*        non-zero, any child Handles contained within the supplied Handle
*        are operated on in the same way as the supplied Handle.
*     rdonly = int (Given)
*        Only used if "oper" is 2. It indicates if the new lock is for
*        read-only or read-write access.
*     result = int * (Returned)
*        Returned holding the result value described under "oper" above.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Returned function value:
*     A pointer to the first Handle that caused the lock or unlock
*     operation to fail. NULL is returned if the operation is successful
*     or faisl for some other reason (in which case status will be set).

*  Description:
*     This function locks or unlocks the supplied Handle for read-only or
*     read-write access. Each Handle is always in one of the following
*     three mutually exclusive states:
*
*     - Unlocked
*     - Locked for read-write access by one and only one thread.
*     - Locked for read-only access by one or more threads.
*
*     When locked for read-write access, the locking thread has exclusive
*     read-write access to the object attached to the supplied handle. When
*     locked for read-only access, the locking thread shares read-only
*     access with zero or more other threads.
*
*     A request to check, lock or unlock a handle can be propagated
*     recursively to all child handles by setting "recurs" non-zero. However,
*     if the attempt to lock or unlock a child fails, no error will be reported
*     and the child will be left unchanged.

*  Notes:
*     - If a thread gets a read-write lock on the handle, and
*     subsequently attempts to get a read-only lock, the existing
*     read-write lock will be demoted to a read-only lock.
*     - If a thread gets a read-only lock on the handle, and
*     subsequently attempts to get a read-write lock, the existing
*     read-only lock will be promoted to a read-write lock only if
*     there are no other locks on the Handle.
*     - "oper" values of -2 and -3 are used in place of 2 and 3 when
*     calling this function recursively from within itself.
*     - A value of zero is returned if an error has already ocurred, or
*     if this function fails for any reason.

*  Authors:
*     DSB: David S Berry (DSB)
*     {enter_new_authors_here}

*  History:
*     10-JUL-2017 (DSB):
*        Initial version
*     19-JUL-2017 (DSB):
*        Added argument rdonly.
*     17-NOV-2017 (DSB):
*        Change the returned values depend on "recurs". Change the API to
*        return a pointer to the first Handle that causes the lock or unlock
*        operation to fail.
*     3-JUL-2018 (DSB):
*        Fix incrementation bugs in loops that loop round lists of
*        read-lockers.
*     {enter_further_changes_here}

*  Copyright:
*     Copyright (C) 2017 East Asian Observatory.
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


#include <pthread.h>
#include "sae_par.h"
#include "dat1.h"
#include "ems.h"
#include "dat_err.h"

/* The initial size for the array holding the identifiers for the threads
   that have a read lock on a handle. It is also the incremement in size
   when the array needs to be extended. */
#define NTHREAD 10

Handle *dat1HandleLock( Handle *handle, int oper, int recurs, int rdonly,
                        int *result, int *status ){

/* Local Variables; */
   Handle *child;
   int ichild;
   int top_level;
   pthread_t *locker;
   pthread_t *rlocker;
   int i;
   int j;
   Handle *error_handle = NULL;
   int child_result;

/* initialise */
   *result = 0;

/* Check inherited status. */
   if( *status != SAI__OK ) return error_handle;

/* To avoid deadlocks, we only lock the Handle mutex for top level
   entries to this function. If "oper" is negative, negate it and set a
   flag indicating we do not need to lock the mutex. */
   if( oper < 0 ) {
      oper = -oper;
      top_level = 0;
   } else {
      top_level = 1;
   }

/* For top-level entries to this function, we need to ensure no other thread
   is modifying the details in the handle, so attempt to lock the handle's
   mutex. */
   if( top_level ) pthread_mutex_lock( &(handle->mutex) );

/* Return information about the current lock on the supplied Handle.
   ------------------------------------------------------------------ */
   if( oper == 1 ) {

/* Default: unlocked */

      if( handle->nwrite_lock ) {
         if( pthread_equal( handle->write_locker, pthread_self() )) {

/* Locked for writing by the current thread. */
            *result = 1;
         } else {

/* Locked for writing by another thread. */
            *result = 2;
         }

      } else if( handle->nread_lock ){

/* Locked for reading by one or more other threads (the current thread does
   not have a read lock on the Handle). */
         *result = 4;

/* Now check to see if the current thread has a read lock, changing the
   above result value if it does. */
         locker = handle->read_lockers;
         for( i = 0; i < handle->nread_lock;i++,locker++ ) {
            if( pthread_equal( *locker, pthread_self() )) {

/* Locked for reading by the current thread (other threads may also have
   a read lock on the Handle). */
               *result = 3;
               break;
            }
         }
      }

/* If required, check any child handles. If we already have a status of
   2, (the supplied handle is locked read-write by another thread), we do
   not need to check the children. */
      if( recurs && *result != 2 ){
         for( ichild = 0; ichild < handle->nchild; ichild++ ) {
            child = handle->children[ichild];
            if( child ) {

/* Get the lock status of the child. */
               (void) dat1HandleLock( child, -1, 1, rdonly, &child_result,
                                      status );

/* If it's 2, we can set the final result and exit immediately. */
               if( child_result == 2 ) {
                  *result = 2;
                  break;

/* Otherwise, ensure the child gives the same result as all the others,
   breaking out and returning the catch-all value if not. */
               } else if(  child_result != *result ) {
                  *result = 5;
                  break;
               }
            }
         }
      }





/* Lock the handle for use by the current thread.
   ------------------------------------------------------------------ */
   } else if( oper == 2 ) {

/* A read-only lock requested.... */
      if( rdonly ) {

/* If the current thread has a read-write lock on the Handle, demote it
   to a read-only lock and return 1 (success). In this case, we know
   there will be no other read-locks. Otherwise if any other thread has
   read-write lock, return zero (failure). */
         if( handle->nwrite_lock ) {
            if( pthread_equal( handle->write_locker, pthread_self() )) {

/* If we do not have an array in which to store read lock thread IDs,
   allocate one now with room for NTHREAD locks. It will be extended as
   needed. */
               if( !handle->read_lockers ) {
                  handle->read_lockers = MEM_CALLOC(NTHREAD,sizeof(pthread_t));
                  if( !handle->read_lockers ) {
                     *status = DAT__NOMEM;
                     emsRep( "", "Could not allocate memory for HDS "
                             "Handle read locks list.", status );
                  }
               }

/* If we now have an array, store the current thread in the first element. */
               if( handle->read_lockers ) {
                  handle->read_lockers[ 0 ] = pthread_self();
                  handle->nread_lock = 1;
                  handle->nwrite_lock = 0;
                  *result = 1;
               }
            }

/* If there is no read-write lock on the Handle, add the current thread
   to the list of threads that currently have a read-only lock, but only
   if it is not already there. */
         } else {

/* Set "result" to 1 if the current thread already has a read-only lock. */
            locker = handle->read_lockers;
            for( i = 0; i < handle->nread_lock;i++,locker++ ) {
               if( pthread_equal( *locker, pthread_self() )) {
                  *result = 1;
                  break;
               }
            }

/* If not, extend the read lock thread ID array if necessary, and append
   the current thread ID to the end. */
            if( *result == 0 ) {
               handle->nread_lock++;
               if( handle->maxreaders < handle->nread_lock ) {
                  handle->maxreaders += NTHREAD;
                  handle->read_lockers = MEM_REALLOC( handle->read_lockers,
                                                    handle->maxreaders*sizeof(pthread_t));
                  if( !handle->read_lockers ) {
                     *status = DAT__NOMEM;
                     emsRep( "", "Could not reallocate memory for HDS "
                             "Handle read locks list.", status );
                  }
               }

               if( handle->read_lockers ) {
                  handle->read_lockers[ handle->nread_lock - 1 ] = pthread_self();

/* Indicate the read-only lock was applied successfully. */
                  *result = 1;
               }
            }
         }

/* A read-write lock requested. */
      } else {

/* If there are currently no locks of any kind, apply the lock. */
         if( handle->nread_lock == 0 ) {
            if( handle->nwrite_lock == 0 ) {
               handle->write_locker = pthread_self();
               handle->nwrite_lock = 1;
               *result = 1;

/* If the current thread already has a read-write lock, indicate success. */
            } else if( pthread_equal( handle->write_locker, pthread_self() )) {
               *result = 1;
            }

/* If there is currently only one read-only lock, and it is owned by the
   current thread, then promote it to a read-write lock. */
         } else if( handle->nread_lock == 1 &&
                    pthread_equal( handle->read_lockers[0], pthread_self() )) {
            handle->nread_lock = 0;
            handle->write_locker = pthread_self();
            handle->nwrite_lock = 1;
            *result = 1;
         }
      }

/* If required, and if the above lock operation was successful, lock any
   child handles that can be locked. */
      if( *result ){
         if( recurs ){
            for( ichild = 0; ichild < handle->nchild; ichild++ ) {
               child = handle->children[ichild];
               if( child ) {
                  error_handle = dat1HandleLock( child, -2, 1, rdonly,
                                                 result, status );
                  if( error_handle ) break;
               }
            }
         }

/* If the lock operation failed, return a pointer to the Handle. */
      } else {
         error_handle = handle;
      }




/* Unlock the handle.
   ----------------- */
   } else if( oper == 3 ) {

/* Assume failure. */
      *result = 0;

/* If the current thread has a read-write lock, remove it. */
      if( handle->nwrite_lock ) {
         if( pthread_equal( handle->write_locker, pthread_self() )) {
            handle->nwrite_lock = 0;
            *result = 1;
         } else {
            *result = -1;
         }

/* Otherwise, if the current thread has a read-only lock, remove it. */
      } else {

/* Loop through all the threads that have read-only locks. */
         locker = handle->read_lockers;
         for( i = 0; i < handle->nread_lock; i++,locker++ ) {

/* If the current thread is found, shuffle any remaining threads down one
   slot to fill the gap left by removing the current thread from the list. */
            if( pthread_equal( *locker, pthread_self() )) {
               rlocker = locker + 1;
               for( j = i + 1; j < handle->nread_lock; j++,locker++ ) {
                  *locker = *(rlocker++);
               }

/* Reduce the number of read-only locks. */
               handle->nread_lock--;
               *result = 1;
               break;
            }
         }
      }

/* If required, and if the above unlock operation was successful, unlock any
   child handles that can be unlocked. */
      if( *result == 1 ){
         if( recurs ){
            for( ichild = 0; ichild < handle->nchild; ichild++ ) {
               child = handle->children[ichild];
               if( child ) {
                  error_handle = dat1HandleLock( child, -3, 1, 0,
                                                 result, status );
                  if( error_handle ) break;
               }
            }
         }

/* If the unlock operation failed, return a pointer to the Handle. */
      } else {
         error_handle = handle;
      }




/* Report an error for any other "oper" value. */
   } else if( *status == SAI__OK ) {
      *status = DAT__FATAL;
      emsRepf( " ", "dat1HandleLock: Unknown 'oper' value (%d) supplied - "
               "(internal HDS programming error).", status, oper );
   }

/* If this is a top-level entry, unlock the Handle's mutex so that other
   threads can access the values in the Handle. */
   if( top_level ) pthread_mutex_unlock( &(handle->mutex) );

/* Return the error handle. */
   return error_handle;
}

