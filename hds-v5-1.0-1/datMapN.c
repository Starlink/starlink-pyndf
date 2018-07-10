/*
*  Name:
*    datMapN

*  Purpose:
*     Map object values a N-dimensional array

*  Language:
*     Starlink ANSI C

*  Invocation:
*     datMapN( HDSLoc* loc, const char * type, const char * mode,
*              int ndim, void **pntr, hdsdim dims[], int * status )

*  Description:
*     This routine maps the primitive object data for reading, writing
*     or updating. The caller is expected to know the number of
*     object dimensions, ndim. The object dimensions are returned
*     in the array, dims[].
*
*     Note that it is not possible to map data of type '_CHAR'.

*  Parameters:
*     loc = HDSLoc * (Given)
*        Locator associated with a structured data object.
*     type = const char * (Given)
*        Expression specifying the data type of the mapped values.
*        If the actual type of the data object differs from this,
*        then conversion will be performed in 'READ' and 'UPDATE'
*        modes.
*     mode = const char * (Given)
*        Expression specifying the mode in which the data are to be
*        mapped.  (Either 'READ', 'WRITE' or 'UPDATE'.)
*     ndim = int (Given)
*        The number of array dimensions allocated to the dims[]
*        array. This must match the actual number of object dimensions.
*     pntr = void** (Returned)
*        Variable to receive the pointer to the memory mapped values.
*     dims[] = hdsdim (Returned)
*        Array to receive the dimensions of the mapped object.
*     status = int * (Given & Returned)
*        Inherited status.

*  Returns:
*     Returns status value on exit.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-09-08 (TIMJ):
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


#include "hds1.h"
#include "dat1.h"
#include "hds.h"
#include "dat_err.h"
#include "ems.h"

#include "sae_par.h"

int datMapN( HDSLoc* loc, const char * type, const char * mode,
	     int ndim, void **pntr, hdsdim dims[], int * status ) {

  int actdim;

  if (*status != SAI__OK) return *status;

  datShape( loc, ndim, dims, &actdim, status );

  if (*status == SAI__OK) {
    if (actdim != ndim) {
      *status = DAT__DIMIN;
      emsRepf( "DAT_MAPN_ERR",
               "Number of dimensions supplied (%d) does not match actual number of dimensions (%d)",
               status, ndim, actdim);
    } else {
      datMap( loc, type, mode, ndim, dims, pntr, status );
    }
  }

  return *status;
}

