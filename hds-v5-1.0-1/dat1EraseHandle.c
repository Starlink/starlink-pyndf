/*
*+
*  Name:
*     dat1EraseHandle

*  Purpose:
*     Recursively erase a handle structure for a component of an HDF object.

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     Handle *dat1EraseHandle( Handle *parent, const char *name, int *status );

*  Arguments:
*     parent = Handle * (Given)
*        Pointer to a Handle that 1) contains the Handle to be erased
*        (if "name" is not NULL), or 2) is the Handle to be erased (if
*        "name" is NULL).
*     name = const char * (Given)
*        The name of the HDF component within "parent" that is to be
*        erased. An error is reported if the named compoent cannot be
*        found.If NULL is supplied, the handle specified by "parent" is
*        itself erased.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Returned function value:
      A NULL pointer is always returned.

*  Description:
*     The named component, and all its sub-components, is erased from the
*     supplied parent handle.

*  Notes:
*     - This function attempts to execure even if an error has already
*     occurred.

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

#include <string.h>

#include "ems.h"
#include "sae_par.h"
#include "dat1.h"
#include "dat_err.h"

Handle *dat1EraseHandle( Handle *parent, const char *name, int * status ){

/* Local Variables; */
   Handle *comp = NULL;
   Handle *child = NULL;
   int ichild;

/* Return immediately no parent is supplied. */
   if( !parent ) return NULL;

/* Find the handle to be erased - either the named component in the
   parent, or the parent itself. */
   if( name ) {
      for( ichild = 0; ichild < parent->nchild; ichild++ ) {
         child = parent->children[ ichild ];
         if( child && child->name && !strcmp( child->name, name ) ) {
            comp = child;

/* We will be erasing this handle ("child") below. The parent will therefore
   no longer have this handle as a child. So remove it from the parent's list
   of children by storing a NULL pointer. */
            parent->children[ ichild ] = NULL;
            break;
         }
      }

   } else {
      comp = parent;
   }

/* If the component was found, attempt to erase all children in the
   component. */
   if( comp ){
      for( ichild = 0; ichild < comp->nchild; ichild++ ) {
         child = comp->children[ ichild ];
         if( child ) {
            if( child->name ){
               dat1EraseHandle( child, NULL, status );
            } else {
               if( *status == SAI__OK ) {
                  *status = DAT__FATAL;
                  emsRepf( " ", "Child handle found with no name inside "
                           "parent '%s' (programming error).", status, name );
               }
            }
         }
      }

/* Then erase the child itself. */
      comp = dat1FreeHandle( comp );
   }

   return NULL;

}


