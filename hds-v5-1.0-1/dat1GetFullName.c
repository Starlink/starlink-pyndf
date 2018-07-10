/*
*+
*  Name:
*     dat1GetFullName

*  Purpose:
*     Get a buffer with the full name of an object

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     name = dat1GetFullName( hid_t objid, int asfile,
*                             ssize_t * namlen, int * status );

*  Arguments:
*     objid = hid_t (Given)
*        Object from which to obtain name/
*     asfile = int (Given)
*        If true obtain the name of the file associated with the
*        object, else obtain the name of the object in the hierarchy.
*     namlen = ssize_t * (Returned)
*        Length of the string in the returned buffer. Can be NULL.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Returned Value:
*     buffer = char *
*        Dynamically allocated buffer containing a nul-terminated
*        string array of the object name. This will be in standard
*        HDF5 form.

*  Description:
*     Determine the name of the HDF5 object by first allocating a
*     bufffer of the correct length and then filling that buffer
*     with the name.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - If the returned value is non-NULL, the memory must
*       be released using MEM_FREE.
*     - The name string will be in HDF5 form as returned by
*       H5Fget_name or H5Iget_name.

*  History:
*     2014-09-03 (TIMJ):
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

#include "hdf5.h"

#include "hds1.h"
#include "dat1.h"

#include "ems.h"
#include "sae_par.h"
#include "dat_err.h"

char *
dat1GetFullName( hid_t objid, int asfile, ssize_t * namlen, int *status) {

  char *tempstr = NULL;
  ssize_t lenstr = 0;
  if (namlen) *namlen = 0;

  if (*status != SAI__OK) return NULL;

  /* Run first to get the size of the buffer we need to use */
  lenstr = (asfile ?
            H5Fget_name(objid, NULL, 0) :
            H5Iget_name(objid, NULL, 0) );
  if (lenstr < 0) {
    *status = DAT__HDF5E;
    dat1H5EtoEMS( status );
    emsRepf("dat1GetFullName_1",
            "Error obtaining length of %s name of locator",
            status, (asfile ? "file" : "path" ) );
    goto CLEANUP;
  }

  /* Allocate buffer of the right length */
  tempstr = MEM_MALLOC( lenstr + 1 );
  if (!tempstr) {
    *status = DAT__NOMEM;
    emsRep( "dat1GetFullName_2", "Malloc error. Can not proceed",
            status);
    goto CLEANUP;
  }

  if (asfile) {
    lenstr = H5Fget_name( objid, tempstr, lenstr+1);
  } else {
    lenstr = H5Iget_name( objid, tempstr, lenstr+1);
  }
  if (lenstr < 0) {
    *status = DAT__HDF5E;
    dat1H5EtoEMS( status );
    emsRepf( "dat1GetFullName_3", "Error obtaining %s name of locator",
             status, (asfile ? "file" : "path") );
    goto CLEANUP;
  }
  if (namlen) *namlen = lenstr;

 CLEANUP:
  if (*status != SAI__OK) {
    if (tempstr) {
      MEM_FREE(tempstr);
      tempstr = NULL;
    }
  }

  return tempstr;


}
