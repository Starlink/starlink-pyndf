#if HAVE_CONFIG_H
#  include <config.h>
#endif

#include <string.h>

#include "ems.h"
#include "hds1.h"
#include "dat1.h"
#include "hds_types.h"
#include "dat_err.h"
#include "hds.h"

#include "sae_par.h"

/*
 *+
 *  Name:
 *    datPut0X

 *  Purpose:
 *    Put a scalar value into an HDS component

 *  Invocation:
 *    status = datPut0X( HDSLoc * loc, <type> value, int * status );

 *  Description:
 *     This routine writes a value into a scalar primitive object.
 *     There is a routine for each access type,
 *
 *        datPut0D    DOUBLE PRECISION
 *        datPut0R    REAL / FLOAT
 *        datPut0I    INTEGER
 *        datPut0K  INT64
 *        datPut0W    WORD / SHORT
 *        datPut0UW   UWORD / unsigned short
 *        datPut0L    LOGICAL
 *        datPut0C    CHARACTER[*n]
 *
 *     If the object data type differs from the access type, then
 *     conversion is performed.
 *
 *     Note that a Vector (1-D) object containing a single value is
 *     different from a Scalar (0-D).

 *  Arguments
 *    HDSLoc * loc = Given
 *       HDS locator associated with a primitive data object.
 *    <type> value = Given
 *       Value to be stored in the primitive data object
 *    int * status = Given & Returned
 *       Global inherited status.

 *  Authors:
 *    Jack Giddings (UCL::JRG)
 *    Sid Wright (UCL::SLW)
 *    Dennis Kelly (REVAD::BDK)
 *    Alan Chipperfield (RAL::AJC)
 *    Tim Jenness (JAC, Hawaii)

 *  History:
 *     3-JAN-1983 (UCL::JRG):
 *       Original.
 *     31-AUG-1983 (UCL::SLW):
 *       Standardise.
 *     05-NOV-1984: (REVAD::BDK)
 *       Remove calls to error system
 *     15-APR-1987 (RAL::AJC):
 *       Improved prologue layout
 *     21-NOV-2005 (TIMJ):
 *       Rewrite in C

 *  Notes:
 *    For datPut0C the supplied string must be nul-terminated.

 *  Copyright:
 *    Copyright (C) 2005 Particle Physics and Astronomy Research Council.
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
 *     Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *     MA 02110-1301, USA

 *  Bugs:
 *     {note_any_bugs_here}

 *-
 */

int datPut0C ( const HDSLoc * loc, const char * value, int * status ) {
  hdsdim dim[DAT__MXDIM];
  int i = 0;
  int ndims;
  int isscalar;

  if ( *status != SAI__OK ) return *status;

  /* Get the rank and dimensions of the object */
  datShape( loc, DAT__MXDIM, dim, &ndims, status );

  /* Check it has only one element (a scalar). */
  isscalar = 1;
  for (i=0; i<ndims; i++) {
    if( dim[i] != 1 ) {
       isscalar = 0;
       break;
    }
  }

  /* Report an error if not. */
  if( !isscalar && *status == SAI__OK ) {
    *status = DAT__DIMIN;
    emsRepf("datPut0C_1", "datPut0C: Data must be scalar.", status );
  }

  datPutC( loc, ndims, dim, value, strlen(value), status );
  return *status;
}

int datPut0D ( const HDSLoc * loc, double value, int * status ) {
  hdsdim dim[DAT__MXDIM];
  int i = 0;
  int ndims;
  int isscalar;

  if ( *status != SAI__OK ) return *status;

  /* Get the rank and dimensions of the object */
  datShape( loc, DAT__MXDIM, dim, &ndims, status );

  /* Check it has only one element (a scalar). */
  isscalar = 1;
  for (i=0; i<ndims; i++) {
    if( dim[i] != 1 ) {
       isscalar = 0;
       break;
    }
  }

  /* Report an error if not. */
  if( !isscalar && *status == SAI__OK ) {
    *status = DAT__DIMIN;
    emsRepf("datPut0D_1", "datPut0D: Data must be scalar.", status );
  }

  datPutD( loc, ndims, dim, &value, status );
  return *status;
}

int datPut0R ( const HDSLoc * loc, float value, int * status ) {
  hdsdim dim[DAT__MXDIM];
  int i = 0;
  int ndims;
  int isscalar;

  if ( *status != SAI__OK ) return *status;

  /* Get the rank and dimensions of the object */
  datShape( loc, DAT__MXDIM, dim, &ndims, status );

  /* Check it has only one element (a scalar). */
  isscalar = 1;
  for (i=0; i<ndims; i++) {
    if( dim[i] != 1 ) {
       isscalar = 0;
       break;
    }
  }

  /* Report an error if not. */
  if( !isscalar && *status == SAI__OK ) {
    *status = DAT__DIMIN;
    emsRepf("datPut0R_1", "datPut0R: Data must be scalar.", status );
  }

  datPutR( loc, ndims, dim, &value, status );
  return *status;
}

int datPut0I ( const HDSLoc * loc, int value, int * status ) {
  hdsdim dim[DAT__MXDIM];
  int i = 0;
  int ndims;
  int isscalar;

  if ( *status != SAI__OK ) return *status;

  /* Get the rank and dimensions of the object */
  datShape( loc, DAT__MXDIM, dim, &ndims, status );

  /* Check it has only one element (a scalar). */
  isscalar = 1;
  for (i=0; i<ndims; i++) {
    if( dim[i] != 1 ) {
       isscalar = 0;
       break;
    }
  }

  /* Report an error if not. */
  if( !isscalar && *status == SAI__OK ) {
    *status = DAT__DIMIN;
    emsRepf("datPut0I_1", "datPut0I: Data must be scalar.", status );
  }

  datPutI( loc, ndims, dim, &value, status );

  return *status;
}

int datPut0K ( const HDSLoc * loc, int64_t value, int * status ) {
  hdsdim dim[DAT__MXDIM];
  int i = 0;
  int ndims;
  int isscalar;

  if ( *status != SAI__OK ) return *status;

  /* Get the rank and dimensions of the object */
  datShape( loc, DAT__MXDIM, dim, &ndims, status );

  /* Check it has only one element (a scalar). */
  isscalar = 1;
  for (i=0; i<ndims; i++) {
    if( dim[i] != 1 ) {
       isscalar = 0;
       break;
    }
  }

  /* Report an error if not. */
  if( !isscalar && *status == SAI__OK ) {
    *status = DAT__DIMIN;
    emsRepf("datPut0K_1", "datPut0K: Data must be scalar.", status );
  }

  datPutK( loc, ndims, dim, &value, status );

  return *status;
}

int datPut0W ( const HDSLoc * loc, short value, int * status ) {
  hdsdim dim[DAT__MXDIM];
  int i = 0;
  int ndims;
  int isscalar;

  if ( *status != SAI__OK ) return *status;

  /* Get the rank and dimensions of the object */
  datShape( loc, DAT__MXDIM, dim, &ndims, status );

  /* Check it has only one element (a scalar). */
  isscalar = 1;
  for (i=0; i<ndims; i++) {
    if( dim[i] != 1 ) {
       isscalar = 0;
       break;
    }
  }

  /* Report an error if not. */
  if( !isscalar && *status == SAI__OK ) {
    *status = DAT__DIMIN;
    emsRepf("datPut0W_1", "datPut0W: Data must be scalar.", status );
  }

  datPutW( loc, ndims, dim, &value, status );

  return *status;
}

int datPut0UW ( const HDSLoc * loc, unsigned short value, int * status ) {
  hdsdim dim[DAT__MXDIM];
  int i = 0;
  int ndims;
  int isscalar;

  if ( *status != SAI__OK ) return *status;

  /* Get the rank and dimensions of the object */
  datShape( loc, DAT__MXDIM, dim, &ndims, status );

  /* Check it has only one element (a scalar). */
  isscalar = 1;
  for (i=0; i<ndims; i++) {
    if( dim[i] != 1 ) {
       isscalar = 0;
       break;
    }
  }

  /* Report an error if not. */
  if( !isscalar && *status == SAI__OK ) {
    *status = DAT__DIMIN;
    emsRepf("datPut0UW_1", "datPut0UW: Data must be scalar.", status );
  }

  datPutUW( loc, ndims, dim, &value, status );

  return *status;
}

int datPut0L ( const HDSLoc * loc, hdsbool_t value, int * status ) {
  hdsdim dim[DAT__MXDIM];
  int i = 0;
  int ndims;
  int isscalar;

  if ( *status != SAI__OK ) return *status;

  /* Get the rank and dimensions of the object */
  datShape( loc, DAT__MXDIM, dim, &ndims, status );

  /* Check it has only one element (a scalar). */
  isscalar = 1;
  for (i=0; i<ndims; i++) {
    if( dim[i] != 1 ) {
       isscalar = 0;
       break;
    }
  }

  /* Report an error if not. */
  if( !isscalar && *status == SAI__OK ) {
    *status = DAT__DIMIN;
    emsRepf("datPut0L_1", "datPut0L: Data must be scalar.", status );
  }

  datPutL( loc, ndims, dim, &value, status );
  return *status;
}
