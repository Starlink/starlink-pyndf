/*
*+
*  Name:
*     dat1H5EtoEMS

*  Purpose:
*     Walk HDF5 error stack and convert to EMS

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     void dat1H5EtoEMS(  int * status );

*  Arguments:
*     status = int* (Given)
*        Pointer to global status.

*  Description:
*     For each entry in the HDF5 error stack, issue a corresponding
*     error using EMS. This ensures that HDF5 errors are visible to
*     Starlink software.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Currently does not show full detail of the entire call stack.
*     - Ensures that dat1InitHDF5 has been called. This prevents
*       HDF5 from issuing its own errors separately.

*  History:
*     2014-08-20 (TIMJ):
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

#include "hdf5.h"

#include "ems.h"
#include "dat1.h"
#include "dat_err.h"

typedef struct HDSH5E_print_t {
  int * status;
} HDSH5E_print_t;

static herr_t
custom_print_cb(unsigned int n, const H5E_error2_t *err_desc, void* client_data);

void dat1H5EtoEMS( int * status ) {

  hid_t curstack;
  HDSH5E_print_t eprint;

  /* Defensive call -- but ensures that we don't get double errors */
  dat1InitHDF5();

  memset( &eprint, 0, sizeof(eprint));
  eprint.status = status;
  curstack = H5Eget_current_stack();
  H5Ewalk( curstack, H5E_WALK_DOWNWARD, custom_print_cb, &eprint );
  H5Eclear2(curstack);

}

/* Callback function for walking the HDF5 error stack */

#define MSG_SIZE 64

/* Heavily influenced by H5E_walk2_cb in H5Eint.c */

static herr_t
custom_print_cb(unsigned int n, const H5E_error2_t *err_desc, void* client_data)
{
  HDSH5E_print_t      *eprint  = (HDSH5E_print_t *)client_data;
  char                maj[MSG_SIZE];
  char                min[MSG_SIZE];
  char                cls[MSG_SIZE];
  int have_desc = 1;
  H5E_major_t maj_num = err_desc->maj_num;
  H5E_minor_t min_num = err_desc->min_num;

  /* Show all information for n==0 case.
     Show description text for all other cases */

  /* We can override the generic status if we have been given
     the generic HDF5 error code. This lets us translate things
     like, "file not found" to the proper DAT__FILNF code. */
  if ( *(eprint->status) == DAT__HDF5E ) {
    if (maj_num == H5E_FILE) {
      if ( min_num == H5E_CANTOPENFILE ) {
        *(eprint->status) = DAT__FILIN;
      }
    }
  }

  /* Get descriptions for the major and minor error numbers */
  H5Eget_class_name(err_desc->cls_id, cls, MSG_SIZE);

  /* Check for "real" error description - used to format output more nicely */
  if(err_desc->desc == NULL || strlen(err_desc->desc) == 0)
    have_desc=0;

  if (n==0) {

    H5Eget_msg(err_desc->maj_num, NULL, maj, MSG_SIZE);
    H5Eget_msg(err_desc->min_num, NULL, min, MSG_SIZE);

    emsRepf("HDF5_INTERNAL_2",
            "%s-DIAG #%03u: %s line %u in %s()%s%s",
            eprint->status,
            cls, n, err_desc->file_name, err_desc->line, err_desc->func_name,
            (have_desc ? ": " : ""),
            (have_desc ? err_desc->desc : "")
            );
    emsRepf("HDF5_INTERNAL_3",
            "%s-DIAG major: %s; minor: %s", eprint->status,
            cls, maj, min );
  } else if (have_desc) {

    emsRepf( "HDF5_INTERNAL_4",
             "%s-DIAG  %s", eprint->status,
             cls, err_desc->desc );

  }


  return 0;

}
