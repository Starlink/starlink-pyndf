/*
*+
*  Name:
*     datLock

*  Purpose:
*     Lock an object for exclusive use by the current thread.

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datLock( HDSLoc *locator, int recurs, int readonly, int *status );

*  Arguments:
*     locator = HDSLoc * (Given)
*        Locator to the object that is to be locked.
*     recurs = int (Given)
*        If the supplied object is locked successfully, and "recurs" is
*        non-zero, then an attempt is made to lock any component objects
*        contained within the supplied object. An error is reported if
*        any components cannot be locked due to them being locked already
*        by a different thread. This operation is recursive - any children
*        of the child components are also locked, etc.
*     readonly = int (Given)
*        If non-zero, the object (and child objects if "recurs" is non-zero)
*        is locked for read-only access. Otherwise it is locked for
*        read-write access.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     This function locks an HDS object for use by the current thread.
*     An object can be locked for read-only access or read-write access.
*     Multiple threads can lock an object simultaneously for read-only
*     access, but only one thread can lock an object for read-write access
*     at any one time. Use of any HDS function that may modify the object
*     will fail with an error unless the thread has locked the object for
*     read-write access. Use of an HDS function that cannot modify the
*     object will fail with an error unless the thread has locked the
*     object (in this case the lock can be either for read-only or
*     read-write access).
*
*     If "readonly" is zero (indicating the current thread wants to
*     modify the object), this function will report an error if any
*     other thread currently has a lock (read-only or read-write) on
*     the object.
*
*     If "readonly" is non-zero (indicating the current thread wants
*     read-only access to the object), this function will report an error
*     only if another thread currently has a read-write lock on the object.
*
*     If the object is a structure, each component object will have its
*     own lock, which is independent of the lock on the parent object. A
*     component object and its parent can be locked by different threads.
*     However, as a convenience function this function allows all
*     component objects to be locked in addition to the supplied object
*     (see "recurs").
*
*     The current thread must unlock the object using datUnlock before it
*     can be locked for use by another thread. All objects are initially
*     locked by the current thread when they are created. The type of
*     access available to the object ("Read", "Write" or "Update")
*     determines the type of the initial lock. For pre-existing objects,
*     this is determined by the access mode specified when calling hdsOpen.
*     For new and temporary objects, the initial lock is always a read-write
*     lock.

*  Notes:
*     - An error will be reported if the supplied object is currently
*     locked by another thread. If "recurs" is non-zero, an error is
*     also reported if any component objects contained within the
*     supplied object are locked by other threads.
*     - The majority of HDS functions will report an error if the object
*     supplied to the function has not been locked for use by the calling
*     thread. The exceptions are the functions that manage these locks -
*     datLock, datUnlock and datLocked.
*     - Attempting to lock an object that is already locked by the
*     current thread will change the type of lock (read-only or
*     read-write) if the lock types differ, but will otherwise have no
*     effect.

*  Authors:
*     DSB: David S Berry (DSB)
*     {enter_new_authors_here}

*  History:
*     10-JUL-2017 (DSB):
*        Initial version
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

#include "ems.h"
#include "sae_par.h"
#include "dat1.h"
#include "hds.h"
#include "dat_err.h"

int datLock( HDSLoc *locator, int recurs, int readonly, int *status ) {

/* Local variables: */
   Handle *error_handle = NULL;
   int lstat;

/* Check inherited status. */
   if (*status != SAI__OK) return *status;

/* Validate input locator. */
   dat1ValidateLocator( "datLock", 0, locator, 0, status );

/* Check we can de-reference "locator" safely. */
   if( *status == SAI__OK ) {

/* Attemp to lock the specified object, plus all its components. If the
   object could not be locked because it was already locked by another
   thread, report an error. */
      error_handle = dat1HandleLock( locator->handle, 2, recurs, readonly, &lstat,
                               status );
      if( error_handle && *status == SAI__OK ) {
         *status = DAT__THREAD;
         emsSetc( "U", readonly ? "read-only" : "read-write" );
         datMsg( "O", locator );
         emsRep( " ", "datLock: Cannot lock HDS object '^O' for ^U use by "
                 "the current thread:", status );
         dat1HandleMsg( "E", error_handle );
         if( error_handle != locator->handle ) {
            emsRep( " ", "A component within it (^E) is locked for writing by another thread.", status );
         } else {
            emsRep( " ", "It is locked for writing by another thread.", status );
         }
      }
   }

   return *status;
}

