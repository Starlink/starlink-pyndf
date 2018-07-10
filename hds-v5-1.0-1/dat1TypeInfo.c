/*
*+
*  Name:
*     dat1TypeInfo

*  Purpose:
*     Obtain information for each data type

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     typeinfo = dat1TypeInfo( );

*  Returned Value:
*     typeinfo = HdsTypeInfo *
*        Structure containing information about each type.

*  Description:
*     Routine returns information about the types.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - BADL is a calculated quantity so can not be stored in
*       a static include file.

*  History:
*     2014-09-15 (TIMJ):
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

#include "hdf5.h"

#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"

#include "prm_par.h"

/* A mutex  used to serialis entry to this function so that multiple
   threads do not try to acces the static data simultaneously. */
static pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;

HdsTypeInfo *
dat1TypeInfo( void ) {

  static int FILLED = 0;
  static HdsTypeInfo typeinfo;

  pthread_mutex_lock( &mutex1 );
  if (!FILLED) {
    size_t i;
    unsigned char * ptr;

    typeinfo.BADD = VAL__BADD;
    typeinfo.BADK = VAL__BADK;
    typeinfo.BADR = VAL__BADR;
    typeinfo.BADI = VAL__BADI;
    typeinfo.BADW = VAL__BADW;
    typeinfo.BADUW = VAL__BADUW;
    typeinfo.BADB = VAL__BADB;
    typeinfo.BADUB = VAL__BADUB;

    typeinfo.BADC = '*';

    /* The bad _LOGICAL value is set to an alternating sequence of zero and one */
    /* bits which is unlikely to occur by accident. It is also made             */
    /* palindromic, so its value does not alter with byte reversal.             */
    ptr = (unsigned char *) &( typeinfo.BADL  );
    for ( i = 0; i < ( ( sizeof( typeinfo.BADL ) + 1 ) / 2 ); i++ ) {
      ptr[ i ] = (unsigned char) ( ( i % 2 ) ? 0x5aU : 0xa5U );
      ptr[ sizeof( typeinfo.BADL ) - i - 1 ] = ptr[ i ];
    }

    FILLED = 1;
  }
  pthread_mutex_unlock( &mutex1 );

  return &typeinfo;
}
