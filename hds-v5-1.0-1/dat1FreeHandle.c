/*
*+
*  Name:
*     dat1FreeHandle

*  Purpose:
*     Free resources used by a handle structure.

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     Handle *dat1FreeHandle( Handle *handle );

*  Arguments:
*     handle = HDSLoc * (Given)
*        Pointer to the Handle to be freed. This function returns without
*        action if NULL is supplied.

*  Returned function value:
*     A NULL pointer is always returned.

*  Description:
*     The memory used by the supplied Handle is freed, and a NULL pointer
*     returned.

*  Notes:
*     - This function will attempt to execute even if an error has
*     already occurred.

*  Authors:
*     DSB: David S Berry (EAO)
*     {enter_new_authors_here}

*  History:
*     5-JUL-2017 (DSB):
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

#include <pthread.h>
#include <string.h>
#include "dat1.h"
#include "ems.h"

Handle *dat1FreeHandle( Handle *handle ) {

/* Return immediately if no Handle was supplied. */
   if( !handle ) return NULL;

/* Free the memory used by components of the Handle structure. */
   if( handle->name ) MEM_FREE( handle->name );
   if( handle->children ) MEM_FREE( handle->children );
   if( handle->read_lockers ) MEM_FREE( handle->read_lockers );

/* Destroy the mutex */
   pthread_mutex_destroy( &(handle->mutex) );

/* Fill the handles with zeros in case any other points to the same
   handle exist. */
   memset( handle, 0, sizeof(*handle) );

/* Free the memory used by the Handle structure itself. */
   MEM_FREE( handle );

/* Return a NULL pointer. */
   return NULL;
}


