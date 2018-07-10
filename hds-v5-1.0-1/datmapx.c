#if HAVE_CONFIG_H
#  include <config.h>
#endif

/*+DATMAP.C-*/

#include <stdio.h>
#include <string.h>
#include "ems.h"      /* EMS error reporting routines            */
#include "hds1.h"     /* Global definitions for HDS              */
#include "dat1.h"     /* Internal dat_ definitions               */
#include "dat_err.h"  /* DAT__ error code definitions            */

#include "hds.h"

/*=============================*/
/* DAT_MAPI - Map INTEGER data */
/*=============================*/
int
datMapI(HDSLoc    *locator,
        const char      *mode_str,
        int       ndim,
        const hdsdim dims[],
        int       **pntr,
        int       *status )
{
#undef context_name
#undef context_message
#define context_name "DAT_MAPI_ERR"
#define context_message\
        "DAT_MAPI: Error mapping HDS primitive as integer values."
   datMap( locator, "_INTEGER", mode_str, ndim, dims,
           (void**)pntr, status );
   return *status;
}

/*=============================*/
/* datMapK - Map INT64 data */
/*=============================*/
int
datMapK(HDSLoc    *locator,
          const char      *mode_str,
          int       ndim,
          const hdsdim dims[],
          int       **pntr,
          int       *status )
{
#undef context_name
#undef context_message
#define context_name "DAT_MAPK_ERR"
#define context_message\
        "DAT_MAPK: Error mapping HDS primitive as 64-bit int values."
   datMap( locator, "_INT64", mode_str, ndim, dims,
           (void**)pntr, status );
   return *status;
}

/*=============================*/
/* DAT_MAPR - Map REAL data */
/*=============================*/
int
datMapR(HDSLoc *locator,
        const char      *mode_str,
        int       ndim,
        const hdsdim dims[],
        float     **pntr,
        int       *status )
{
#undef context_name
#undef context_message
#define context_name "DAT_MAPR_ERR"
#define context_message\
        "DAT_MAPR: Error mapping an HDS primitive as real values."
   datMap( locator, "_REAL", mode_str, ndim, dims,
           (void**)pntr, status );
   return *status;
}

/*======================================*/
/* DAT_MAPD - Map DOUBLE PRECISION data */
/*======================================*/
int
datMapD(HDSLoc *locator,
        const char      *mode_str,
        int       ndim,
        const hdsdim dims[],
        double    **pntr,
        int       *status )
{
#undef context_name
#undef context_message
#define context_name "DAT_MAPD_ERR"
#define context_message\
        "DAT_MAPD: Error mapping an HDS primitive as double precision values."
   datMap( locator, "_DOUBLE", mode_str, ndim, dims,
           (void**)pntr, status );
   return *status;
}

/*=============================*/
/* DAT_MAPL - Map LOGICAL data */
/*=============================*/
int
datMapL(HDSLoc *locator,
        const char      *mode_str,
        int       ndim,
        const hdsdim dims[],
        int       **pntr,
        int       *status )
{
#undef context_name
#undef context_message
#define context_name "DAT_MAPL_ERR"
#define context_message\
        "DAT_MAPL: Error mapping an HDS primitive as logical values."
   datMap( locator, "_LOGICAL", mode_str, ndim, dims,
           (void**)pntr, status );
   return *status;
}

/*===============================*/
/* DAT_MAPC - Map CHARACTER data */
/*===============================*/
int
datMapC(HDSLoc *locator,
        const char      *mode_str,
        int       ndim,
        const hdsdim dims[],
        unsigned char **pntr,
        int       *status )
{
#undef context_name
#undef context_message
#define context_name "DAT_MAPC_ERR"
#define context_message\
        "DAT_MAPC: Error mapping an HDS primitive as character values."
   datMap( locator, "_CHAR", mode_str, ndim, dims,
           (void**)pntr, status );
   return *status;
}


/*===============================*/
/* DAT_MAPV - Map values associated with an object as if vectorized */
/*===============================*/
int
datMapV(HDSLoc *locator,
	const char      *type_str,
        const char      *mode_str,
        void      **pntr,
	size_t    *actval,
        int       *status )
{
#undef context_name
#undef context_message
#define context_name "DAT_MAPV_ERR"
#define context_message\
        "DAT_MAPV: Error mapping an HDS vectorized primitive."

  /* Local variables */
  hdsdim dims[DAT__MXDIM];
  int    ndim;

  /* Initialise return values */
  *pntr = NULL;
  *actval = 0;

  datSize( locator, actval, status );
  datShape( locator, DAT__MXDIM, dims, &ndim, status );
  datMap( locator, type_str, mode_str, ndim, dims,
	  pntr, status );
  return *status;
}


