/*
*+
*  Name:
*     dat1HandleMsg

*  Purpose:
*     Create an MSG token holding the full path of a Handle.

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     dat1HandleMsg( const char *token, const Handle *handle ){

*  Arguments:
*     token = const char * (Given)
*        The name of the MSG message tokento create.
*     handle = const Handle * (Given)
*        Pointer to the Handle.

*  Description:
*     This function creates an HDS path from the names stores in the
*     supplied Handle structure and its parents, and assigns the path to
*     the specified message token.

*  Notes:
*     - This function attempts to execute even if an error has already
*     occurred.

*  Authors:
*     DSB: David S Berry (EAO)
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

#define MAX_DEPTH 100

void dat1HandleMsg( const char *token, const Handle *handle ){

/* Local Variables; */
   Handle *parent;
   const char *names[ MAX_DEPTH ];
   int i;

/* Return immediately if no message token was supplied. */
   if( !token ) return;

/* Check a Handle was supplied. */
   if( handle ) {

/* Work up the tree from the supplied Handle to the root, saving pointers
   to the name strings at each level. */
      i = 0;
      names[ i ] = handle->name;
      parent = handle->parent;
      while( parent && ++i < MAX_DEPTH ) {
         names[ i ] = parent->name;
         parent = parent->parent;
      }

/* If we exceeded the maximum depth, prefix the returned HDS path with an
   ellipsis to indicate that some leading fields have been omitted. */
      if( i == MAX_DEPTH ) {
         emsSetc( token, "..." );
         i--;
      }

/* Work down the tree from the root, concatenating the names into a
   message token. */
      for( ; i > 0; i-- ) {
         emsSetc( token, names[ i ] );
         emsSetc( token, "." );
      }
      emsSetc( token, names[ 0 ] );

/* If no Handle was supplied, return a blacnk token. */
   } else {
      emsSetc( token, " " );
   }
}


