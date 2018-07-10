/*
*+
*  Name:
*     dat1ValidateLocator

*  Purpose:
*     Check the supplied locator is usable.

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     dat1ValidateLocator( const char *func, int checklock, const HDSLoc *loc,
*                          int rdonly, int * status );

*  Arguments:
*     func = const char * (Given)
*        Name of calling function. Used in error messages.
*     checklock = int (Given)
*        If non-zero, an error is reported if the supplied locator is not
*        locked by the current thread (see datLock). This check is not
*        performed if "checklock" is zero.
*     loc = HDSLoc * (Given)
*        Locator to validate.
*     rdonly = int (Given)
*        Indicates if the calling function may or may not make any
*        changes to the HDF object or locator structure. If a non-zero
*        value is supplied, it is assumed that the calling function will
*        never make any changes to either the HDF object on disk or the
*        locator structure. This determines the type of lock that the
*        calling thread must have on the object (read-only or read-write)
*        to avoid an error being reported by this function.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     An error is reported if the supplied locator is not valid. This can
*     occur for instance if the supplied locator is a secondary locator
*     that has been annulled automatically as a result of the file being
*     closed. An error is also reported if the current thread does no
*     have an appropriate lock on the supplied object.

*  Authors:
*     DSB: David Berry (EAO)
*     {enter_new_authors_here}

*  History:
*     7-JUL-2017 (DSB):
*        Initial version
*     {enter_further_changes_here}

*  Copyright:
*     Copyright (C) 2017 East Asian Observatory
*     All Rights Reserved.

*  Licence:
*     This program is free software; you can redistribute it and/or
*     modify it under the terms of the GNU General Public License as
*     published by the Free Software Foundation; either version 3 of
*     the License, or (at your option) any later version.
*
*     This program is distributed in the hope that it will be
*     useful, but WITHOUT ANY WARRANTY; without even the implied
*     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
*     PURPOSE. See the GNU General Public License for more details.
*
*     You should have received a copy of the GNU General Public License
*     along with this program; if not, write to the Free Software
*     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
*     MA 02110-1301, USA.

*  Bugs:
*     {note_any_bugs_here}
*-
*/

#include <pthread.h>
#include "sae_par.h"
#include "ems.h"
#include "dat1.h"
#include "hds1.h"
#include "hds.h"
#include "dat_err.h"

int dat1ValidateLocator( const char *func, int checklock, const HDSLoc *loc,
                         int rdonly, int * status ) {

/* Local Variables; */
   int valid;
   int lock_status;

/* If the locator has been annulled (e.g. due to the container file being
   closed when the last primary locator was annulled), report an error. */
   datValid( loc, &valid, status );
   if( !valid && *status == SAI__OK ) {
      *status = DAT__LOCIN;
      emsRepf(" ", "%s: The supplied HDS locator is invalid - it may have been "
             "annulled as a result of the associated file being closed.",
             status, func );
   }

/* Report an error if there is no handle in the locator. */
   if( loc && !loc->handle && *status == SAI__OK ) {
      *status = DAT__FATAL;
      datMsg( "O", loc );
      emsRepf( " ", "%s: The supplied HDS locator for '^O' has no handle (programming error).",
              status, func );
   }

/* If required, check that the object is locked by the current thread for
   the appropriate type of access. Do not check any child objects as these
   will be checked if and when accessed. */
   if( checklock && *status == SAI__OK && loc->handle->docheck ) {
      dat1HandleLock( loc->handle, 1, 0, 0, &lock_status, status );

/* Calling function will not make any change to the object. In this case
   the current thread must have a lock but the type (read-only or
   read-write) does not matter. */
      if( rdonly ) {
         if( lock_status != 1 && lock_status != 3 && *status == SAI__OK ) {
            *status = DAT__THREAD;
            datMsg( "O", loc );
            emsRepf( " ", "%s: The supplied HDS locator for '^O' cannot be used.",
                    status, func );
            emsRep( " ", "It has not been locked for read-only or read-write "
                    "access by the current thread (programming error).",
                    status );
         }

/* Calling function may make changes to the object. In this case the
   current thread must have a read-write lock. */
      } else if( lock_status != 1 && *status == SAI__OK ) {
         *status = DAT__THREAD;
         datMsg( "O", loc );
         emsRepf( " ", "%s: The supplied HDS locator for '^O' cannot be used.",
                 status, func );
         if( lock_status == 3 ) {
            emsRep( " ", "Write-access is not allowed (the current thread "
                    "has locked it for read-only access - programming error).",
                    status );
         } else {
            emsRep( " ", "It has not been locked for read-write access by the "
                    "current thread (programming error).", status );
         }
      }
   }

   return *status;

}
