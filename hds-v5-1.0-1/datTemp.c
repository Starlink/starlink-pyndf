/*
*+
*  Name:
*     datTemp

*  Purpose:
*     Create temporary object

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     datTemp( const char *type_str, int ndim, const hdsdim dims[],
*              HDSLoc **locator, int *status );

*  Arguments:
*     type_str = const char * (Given)
*        Data type.
*     ndim = int (Given)
*        Number of dimensions.
*     dims = const hdsdim [] (Given)
*        Object dimensions.
*     locator = HDSLoc ** (Returned)
*        Temporary object locator.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Create an object that exists only for the lifetime of the
*     program run. This may be used to hold temporary objects -
*     including those mapped to obtain scratch space.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - If type matches one of the primitive type names, a primitive of
*       appropriate type is created; otherwise the object is assumed to
*       be a structure. If the object is a structure array, loc will be
*       associated with the complete array, not the first cell. Thus,
*       new components can only be created through another locator which
*       is explicitly associated with an individual cell (see datCell).

*  History:
*     2014-10-16 (TIMJ):
*        Initial version
*     2014-10-28 (TIMJ):
*        First working version. Reuses the locator so not sure
*        what happens if you create a primitive top level and then
*        ask for a structure. May well need to create an extra layer
*        of hierarchy, store the locator to the top-level but return
*        a locator from a level below with a dynamic name.
*     2014-11-14 (TIMJ):
*        Now create a root HDS_SCRATCH locator and for each call to the
*        routine create a TEMP_nnn object for the caller. This is how
*        HDSv4 implemented things.
*     2017-9-8 (DSB):
*        Keep the file name (with suffix) in a static variable so that we
*        can use it to unlick the file on subsequent invocations of this
*        function.
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

#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#include "ems.h"
#include "sae_par.h"
#include "star/one.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"
#include "dat_par.h"
#include "dat_err.h"

/* Mutex used to serialise access to the following static variables */
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static HDSLoc *tmploc = NULL;
static size_t tmpcount = 0;
static char fname_with_suffix[256+DAT__SZFLX];

int
datTemp( const char *type_str, int ndim, const hdsdim dims[],
         HDSLoc **locator, int *status ) {

  char * prefix = NULL;
  char fname[256];
  char tempname[DAT__SZNAM+1];
  hdsbool_t there = 1;
  int init;

  if (*status != SAI__OK) return *status;

  /* Since this function uses global variables, we serialise access to it
     by requiring each thread to acquire a mutex lock before proceeding. */
  pthread_mutex_lock( &mutex );

  /* We create one temporary file per process. We create a top-level
     container object and for each call to datTemp we create a new
     structure of the requested type and dimensionality. We have to
     create this extra layer to enforce a namespace on temporary
     structures (and otherwise some one creating a primitive temp
     type will mess up subsequent calls). Note that the temp root locator
     therefore lives for as long as the process as there is no API to
     annul the locator that we cache. */

  /* Create a temp file if required */
  if (!tmploc) {

    /* Probably should use the OS temp file name generation
       system -- but for now use the HDS scheme. */
    prefix = getenv( "HDS_SCRATCH" );
    one_snprintf( fname, sizeof(fname), "%s/t%x", status,
                  (prefix ? prefix : "."), getpid() );

    /* Open the temp file: type and name are the same. The returned
       locator is locked by the current thread. */
    hdsNew(fname, "HDS_SCRATCH", "HDS_SCRATCH", 0, dims, &tmploc, status );

    /* Get the name of the file with suffix. Store in a static variable so
       that we can access it later in this function on subseuqnent
       invocations. */
    one_snprintf(fname_with_suffix, sizeof(fname_with_suffix),"%s%s", status,
                 fname, DAT__FLEXT);
  }

  /* Lock the container file for read-write access by the current thread. */
  datLock( tmploc, 0, 0, status );

  /* Create a structure inside the temporary file. Compatibility with HDS
     suggests we call these TEMP_nnnn (although we only have to use the
     scheme that hdsInfoI is expecting. HDS used a global temp counter as
     a starting point as that gives you an idea of the total number of
     temp components that have been created and there is no reasonable
     chance of running out of counter space */

  do {
    one_snprintf(tempname, sizeof(tempname), "TEMP_%-*zu", status,
                 (int)(sizeof(tempname) - 1 - 5), ++tmpcount );
    datThere(tmploc, tempname, &there, status ); /* multi-threaded race here... */
    if (*status != SAI__OK) break;
  } while (there);

  /* Now create the temporary object of the correct type and size */
  *locator = dat1New( tmploc, 0, tempname, type_str, ndim, dims, status );

  /* Unlock the container file so that other threads can create temporary
     objects in it. */
   datUnlock( tmploc, 0, status );

  /* Usually at this point you should unlink the file and hope the
     operating system will keep the file handle open whilst deferring the delete.
     This will work on unix systems. On Windows not so well. */
  if (*status == SAI__OK) unlink(fname_with_suffix);

  /* Unlock the mutex. */
  pthread_mutex_unlock( &mutex );

  return *status;
}
