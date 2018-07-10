/*
*+
*  Name:
*     

*  Purpose:
*     

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     dat1EncodeSubscript( int ndim, hdsbool_t canbecell,
*                          const hdsdim lower[], const hdsdim upper[],
*                          char *buf, size_t buflen, int *status );

*  Arguments:
*     ndim = int (Given)
*        Number of dimensions.
*     canbecell = hdsbool_t (Given)
*        If true, coordinates can be treated as a single cell
*        and written as such, if lower==upper for all dimensions.
*        Ignored if upper is NULL.
*     lower = const hdsdim [] (Given)
*        Lower bounds of subsection.
*     upper = const hdsdim [] (Given)
*        Upper bounds of sub section. Should be NULL for an array cell.
*     buf = char * (Returned)
*        Buffer to receive encoded subscript expression.
*     buflen = size_t (Given)
*        Allocated length of "buf".
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Given lower and upper bounds of a hyperslab, convert that to the
*     standard HDS subscript expression. "(n,n,...)" for an array cell,
*     or "(l:u,l:u,...)" for a slice.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - Expects another routine to extract the bounds from a dataspace hyperslab.
*     - Can not tell if the bounds cover the full data array. Do not call this
*       routine if this is representing the full array.

*  History:
*     2014-09-09 (TIMJ):
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

#include "ems.h"
#include "sae_par.h"
#include "star/one.h"
#include "prm_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

void
dat1EncodeSubscript( int ndim, hdsbool_t canbecell, const hdsdim lower[], const hdsdim upper[],
                     char *buf, size_t buflen, int *status ) {
  int i;
  hdsbool_t iscell = 0;

  if (*status != SAI__OK) return;

  buf[0] = '\0';
  one_strlcpy( buf, "(", buflen, status );

  if (!upper) {
    iscell = 1;
  } else if (canbecell) {
    iscell = 1;
    for (i=0; i<ndim; i++) {
      if (lower[i] != upper[i]) {
        iscell = 0;
        break;
      }
    }
  }


  for (i=0; i<ndim; i++) {
    char coordstr[VAL__SZK+1];
    char ucoordstr[VAL__SZK+1];
    if (upper && !iscell) {
      one_snprintf(ucoordstr, sizeof(ucoordstr),
                   ":%zu", status, (size_t)upper[i] );
    }
    one_snprintf(coordstr, sizeof(coordstr), "%zu%s%s",
                 status, (size_t)lower[i],
                 ((upper && !iscell) ? ucoordstr : ""),
                 (ndim-i==1 ? "" : ","));
    one_strlcat( buf, coordstr, buflen, status );
  }

  one_strlcat( buf, ")", buflen, status );
}
