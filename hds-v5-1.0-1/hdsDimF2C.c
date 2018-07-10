/*
*+
*  Name:
*    hdsDimF2C

*  Purpose:
*    Convert an array of F77_INTEGER_TYPE[] to hdsdim[]

*  Invocation:
*    outdims = hdsDimF2C( int ndim, const F77_INTEGER_TYPE fdims[],
*                         hdsdim cdims[DAT__MXDIM], int * status );

*  Description:
*    This function should be used to convert a an array of Fortran dimensions
*    of type F77_INTEGER_TYPE to an array of dimensions suitable for HDS C
*    usage. Returns a pointer to an array of hdsdim[] suitable
*    for the C function.

*  Arguments
*    ndim = int (Given)
*       Number of relevant dimensions. Should not exceed DAT__MXDIM.
*    fdims[DAT__MXDIM] = const F77_INTEGER_TYPE (Given)
*       Input dimensions to copy, of size ndim.
*    cdims[] = hdsdim (Given)
*       Buffer space that can be used to store the copied dimensions.
*       Note that there is no guarantee that at exit this array will
*       have been used.
*    int *status = Given and Returned
*       Inherited status. If set, this routine will return NULL.

*  Return Value:
*    outdims = hdsdim*
*       Pointer to an array of HDS C integers containing the dimensions.

*  Authors:
*    Tim Jenness (JAC, Hawaii)
*    Tim Jenness (Cornell University)

*  History:
*    12-JUL-2005 (TIMJ):
*      Initial version
*    2014-09-15 (TIMJ):
*      For HDS-H5

*  Notes:
*    - Only use the pointer returned by this routine. Do not
*      assume that cdims[] will be filled since it may not be
*      used if the type of hdsdim is the same as a F77_INTEGER_TYPE.
*    - The expectation is that this routine is used solely for C
*      interfaces to Fortran library routines or converting from Fortran
*      to C.
*    - A Fortran INTEGER will always fit in a hdsdim without overflow.

*  Copyright:
*    Copyright (C) 2014 Cornell University.
*    Copyright (C) 2006 Particle Physics and Astronomy Research Council.
*    All Rights Reserved.

*  Licence:
*     This program is free software; you can redistribute it and/or
*     modify it under the terms of the GNU General Public License as
*     published by the Free Software Foundation; either version 2 of
*     the License, or (at your option) any later version.
*
*     This program is distributed in the hope that it will be
*     useful, but WITHOUT ANY WARRANTY; without even the implied
*     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
*     PURPOSE. See the GNU General Public License for more details.
*
*     You should have received a copy of the GNU General Public
*     License along with this program; if not, write to the Free
*     Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*     MA 02111-1307, USA

*  Bugs:
*     {note_any_bugs_here}

*-
*/

#if HAVE_CONFIG_H
#  include <config.h>
#endif

#include <stdlib.h>
#include <string.h>

#include "ems.h"
#include "star/mem.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds_types.h"
#include "dat_err.h"

#include "hds1_types.h"
#include "hds_fortran.h"

hdsdim *
hdsDimF2C( int ndim, const F77_INTEGER_TYPE fdims[],
	   hdsdim cdims[DAT__MXDIM], int * status ) {

#if HDS_COPY_FORTRAN_DIMS
  int i;   /* loop counter */
#endif
  hdsdim * retval = NULL;

  if ( *status != SAI__OK ) return NULL;

#if HDS_COPY_FORTRAN_DIMS
  /* sizes or signs differ so we need to copy one at a time
     and cast to the new type */

  for (i = 0; i < ndim; i++ ) {
      cdims[i] = (hdsdim)fdims[i];
  }

  /* check status is good before deciding to use this array */
  if (*status == SAI__OK) retval = cdims;

#else
  /* hdsdim is the same size and sign so no copy required */
  retval = (hdsdim*)fdims;
#endif

  return retval;

}

