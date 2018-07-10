/*
*+
*  Name:
*     datLocked

*  Purpose:
*     See if an object is locked.

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     locked = datLocked( const HDSLoc *locator, int recursive, int *status );

*  Arguments:
*     locator = const HDSLoc * (Given)
*        A locator for the object to be checked.
*     recursive = int (Given)
*        If non-zero, then all descendants of the supplied object are
*        also checked in the same way.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Returned function value:
*     A value indicating the status of the supplied Object:
*
*    -1: the application is is linked with a version of HDS that does
*        not support object locking.
*
*     0: the supplied object is unlocked. If "recursive" is non-zero,
*        then all descendant objects are also unlocked, and this is then
*        the condition that must be met for the current thread to be able
*        to lock the supplied object for read-write access using function
*        datLock. This condition can be achieved by releasing any existing
*        locks using datUnlock.
*
*     1: the supplied object is locked for reading and writing by the
*        current thread. This is the condition that must be met for the current
*        thread to be able to use the supplied object in any HDS function
*        that might modify the object (except for the locking and unlocking
*        functions - see below). If "recursive" is non-zero, then all
*        descendant objects are also locked for reading and writing. This
*        condition can be achieved by calling datLock.
*
*     2: the supplied object is locked for reading and writing by a different
*        thread. An error will be reported if the current thread attempts to
*        use the object in any other HDS function. If "recursive" is non-zero,
*        then either the object itself or one of its descendant objects is
*        locked for reading and writing.
*
*     3: the supplied object is locked read-only by the current thread
*        (and maybe other threads as well). If "recursive" is non-zero,
*        then all descendant objects are also locked read-only by the
*        current thread. This is the condition that must be met for the
*        current thread to be able to use the supplied object in any HDS
*        function that cannot modify the object. An error will be
*        reported if the current thread attempts to use the object in any
*        HDS function that could modify the object. This condition can be
*        achieved by calling datLock.
*
*     4: the supplied object is not locked by the current thread, but is
*        locked read-only by one or more other threads. An error will be
*        reported if the current thread attempts to use the object in any
*        other HDS function. If "recursive"  is non-zero, then all
*        descendant objects are also locked read-only by one or more
*        other threads.
*
*     5: Some complex mix of locked and unlocked descendants not covered by
*        any of the above values.

*  Description:
*     This function returns a value that indicates if the object
*     specified by the supplied locator has been locked for use by one or
*     more threads. A thread can lock an object either for read-only
*     access or for read-write access. The lock management functions
*     (datLock and datUnlock) will ensure that any thread that requests
*     and is given a read-write lock will have exclusive access to the
*     object - no other locks of either type will be issued to other
*     threads until the first thread releases the read-write lock using
*     datUnlock. If a thread requests and is given a read-only lock, the
*     lock management functions may issue read-only locks to other
*     threads, but it will also ensure that no other thread is granted
*     a read-write lock until all read-only locks have been released.

*  Notes:
*     - The locking performed by datLock, datUnlock and datLocked is
*     based on POSIX threads, and has no connection with the locking
*     referred to in hdsLock and hdsFree.
*     - Zero is returned as the function value if an error has already
*     occurred, or if an error occurs in this function.

*  Authors:
*     DSB: David S Berry (DSB)
*     {enter_new_authors_here}

*  History:
*     7-JUL-2017 (DSB):
*        Initial version
*     17-NOV-2017 (DSB):
*        Added argument "recursive".
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
#include "hds.h"

int datLocked( const HDSLoc *locator, int recursive, int *status ) {

/* Local Variables; */
   int result = 0;

/* Check inherited status. */
   if (*status != SAI__OK) return result;

/* Validate input locator, but do not include the usual check that the
   object is locked by the current thread since we'll be performing that
   test as part of this function. */
   dat1ValidateLocator( "datLocked", 0, locator, 1, status );

/* Get the value to return. Test status first so that we know it is safe
   to deference "locator". */
   if( *status == SAI__OK ) {
      dat1HandleLock( locator->handle, 1, recursive, 0, &result, status );
   }

/* Return the result. */
   return result;
}

