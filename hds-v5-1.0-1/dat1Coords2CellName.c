/*
*+
*  Name:
*     dat1Coords2CellName

*  Purpose:
*     Given coordinates, determine the name of the "cell"

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     void dat1Coords2CellName( int ndim, const hdsdim coords[], char * cellname,
*                               size_t cellnamelen, int * status );

*  Arguments:
*     ndim = int (Given)
*        Number of dimensions in coords.
*     coords = const hdsdim [] (Given)
*        Coordinates of the cell.
*     cellname = char * (Returned)
*        String buffer to receive the cell name
*     cellnamelen = size_t (Given)
*        Allocated lengh of cellname
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Convert the supplied cell coordinates into the corresponding name of
*     the HDF5 Group that should be selected.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     - No attempt is made to bounds check the supplied coordinates
*       to ensure that they are in a valid range.

*  History:
*     2014-09-05 (TIMJ):
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
#include "sae_par.h"
#include "star/one.h"
#include "prm_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

void
dat1Coords2CellName( int ndim, const hdsdim coords[], char * cellname,
                     size_t cellnamelen, int * status ) {

  const char nameroot[] = DAT__CELLNAME;
  size_t lenstr = 0;

  if (*status != SAI__OK) return;

  /* Now format the coordinates into the name, worrying
     about when to include the comma */
  one_strlcpy( cellname, nameroot, cellnamelen, status );
  lenstr = strlen(cellname);

  dat1EncodeSubscript(ndim, 1, coords, NULL, &(cellname[lenstr]),
                      cellnamelen - lenstr, status );

}
