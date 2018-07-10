#include <pthread.h>
#include <string.h>

#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

/* Variable storing tuned state */

/* Indicates that we have looked at the environment */
static hdsbool_t HAVE_INITIALIZED_V5_TUNING = 0;

/* These are all the parameters that can be tuned along
   with their defaults. */

static hds_shell_t HDS_SHELL = HDS__SHSHELL; /* Default to doing expansion */

/* Should memory mapping be enabled: 1 (yes), 0 (no) */

static hdsbool_t HDS_MAP = HDS_TRUE; /* Do mmap by default when possible */

/* A mutex used to serialise access to the getters and setters so that
   multiple threads do not try to access the global data simultaneously. */
static pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
#define LOCK_MUTEX pthread_mutex_lock( &mutex1 );
#define UNLOCK_MUTEX pthread_mutex_unlock( &mutex1 );

/* Parse tuning environment variables. Should only be called once the
   first time a tuning parameter is required */

static void hds1SetShell( hds_shell_t shell);
static void hds1SetUseMmap( hdsbool_t use_mmap );

static void hds1ReadTuneEnvironment () {
  int itemp = 0;
  if (HAVE_INITIALIZED_V5_TUNING) return;

  /* dat1Getenv solely knows about environment variables
     with integers and not about range checking so we do the range check
     here. */

  itemp = HDS_SHELL;
  dat1Getenv( "HDS_SHELL", itemp, &itemp );
  hds1SetShell( itemp );

  itemp = (HDS_MAP ? 1 : 0);
  dat1Getenv( "HDS_MAP", HDS_MAP, &itemp );
  hds1SetUseMmap( itemp ? HDS_TRUE : HDS_FALSE );

  HAVE_INITIALIZED_V5_TUNING = 1;
}


/*
*+
*  Name:
*     hdsTune

*  Purpose:
*     Set HDS tuning parameter

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hdsTune(const char *param_str, int  value, int  *status);

*  Arguments:
*     param_str = const char * (Given)
*        Name of the tuning parameter.
*     value = int (Given)
*        New parameter value.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Alter an HDS control setting.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Supports MAP and SHELL tuning parameters
*     - Other HDS Classic tuning parameters are ignored.

*  History:
*     2014-09-10 (TIMJ):
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

int
hdsTune(const char *param_str, int  value, int  *status) {

  if (*status != SAI__OK) return *status;

  /* HDS supports options:
     - MAP: Mapping mode
     - INAL: Initial file allocation
     - 64BIT: 64-bit mode (ie HDSv4, else HDSv3)
     - MAXW: Working page list
     - NBLOCKS: Size of internal transfer buffer
     - NCOM: Optimum number of structure components
     - SHELL: Shell used for name expansion
     - SYSL: System wide locking flag
     - WAIT: Wait for locked files

     INAL, MAXW, NBLOCKS, NCOM, SYSL and WAIT are all irrelevant.

     64BIT will have no effect as we are using whatever HDF5 gives us.

     Only currently support SHELL and MAP.

     Ignore the ones that are irrelevant. Warn about those that
     might become relevant.

  */

  if (strncmp( param_str, "INAL", 4 ) == 0 ||
      strncmp( param_str, "64BIT", 5 ) == 0 ||
      strncmp( param_str, "MAXW", 4 ) == 0 ||
      strncmp( param_str, "NBLO", 4 ) == 0 ||
      strncmp( param_str, "NCOM", 4 ) == 0 ||
      strncmp( param_str, "SYSL", 4 ) == 0 ||
      strncmp( param_str, "WAIT", 4 ) == 0 ) {
    /* Irrelevant for HDF5 */
  } else if (strncmp( param_str, "MAP", 3) == 0 ) {
    hds1SetUseMmap( value ? HDS_TRUE : HDS_FALSE );
  } else if (strncmp( param_str, "SHEL", 4) == 0) {
    hds1SetShell( value );
  } else {
    *status = DAT__NAMIN;
    emsRepf("hdsTune_1", "hdsTune: Unknown tuning parameter '%s'",
            status, param_str );
  }

  return *status;
}

/*
*+
*  Name:
*     hdsGtune

*  Purpose:
*     Obtain tuning parameter value

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hdsGtune(const char *param_str, int *value, int *status);

*  Arguments:
*     param = const char * (Given)
*        Name of the tuning parameter whose value is required (case insensitive).
*     value = int * (Returned)
*        Current value of the parameter.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     The routine returns the current value of an HDS tuning parameter
*     (normally this will be its default value, or the value last
*     specified using the hdsTune routine).

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Supports MAP and SHELL options.
*     - The SHELL tuning parameter does not use public
*       constants but declares that (-1=no shell, 0=sh, 2=csh, 3=tcsh).
*       This implementation only understands -1 and 0.
*     - Tuning parameters may be abbreviated to 4 characters.

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

int
hdsGtune(const char *param_str, int *value, int *status) {

  if (*status != SAI__OK) return *status;

  if (strncasecmp(param_str, "SHEL", 4) == 0) {
    *value = hds1GetShell();
  } else if (strncasecmp(param_str, "MAP", 3) == 0) {
    *value = hds1GetUseMmap();
  } else {
    *status = DAT__NOTIM;
    emsRep("hdsGtune", "hdsGtune: Not yet implemented for HDF5",
           status);
  }
  return *status;
}

/* Getter and setter routines for internal use */

hdsbool_t hds1GetUseMmap() {
  hdsbool_t result;
  /* Ensure that defaults have been read */
  hds1ReadTuneEnvironment();
  LOCK_MUTEX;
  result = HDS_MAP;
  UNLOCK_MUTEX;
  return result;
}

static void hds1SetUseMmap( hdsbool_t use_mmap ) {
  LOCK_MUTEX
  HDS_MAP = use_mmap;
  UNLOCK_MUTEX
  return;
}

hds_shell_t hds1GetShell() {
  hds_shell_t result;
  /* Ensure that defaults have been read */
  hds1ReadTuneEnvironment();
  LOCK_MUTEX;
  result = HDS_SHELL;
  UNLOCK_MUTEX;
  return result;
}

static void hds1SetShell( hds_shell_t shell) {
  /* Range check -- revert to SHSHELL if out of range */
  LOCK_MUTEX
  if (shell >= HDS__NOSHELL && shell < HDS__MAXSHELL) {
    HDS_SHELL = shell;
  } else {
    HDS_SHELL = HDS__SHSHELL;
  }
  UNLOCK_MUTEX
  return;
}
