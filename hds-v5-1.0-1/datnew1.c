#if HAVE_CONFIG_H
#  include <config.h>
#endif

/*+DATNEW.C-*/

/* Include files */
#include <stdio.h>                  /* stdio for sprintf() prototype         */

#include "ems.h"                    /* EMS error reporting routines          */

#include "hds1.h"                   /* Global definitions for HDS            */
#include "dat1.h"                   /* Internal dat_ definitions             */
#include "dat_err.h"                /* DAT__ error code definitions          */
#include "hds.h"
#include "hds_types.h"
#include "sae_par.h"

/*====================================*/
/* DAT_NEWC - Create string component */
/*====================================*/

int
datNewC(const HDSLoc    *locator,
        const char      *name_str,
        size_t    len,
        int       ndim,
        const hdsdim dims[],
        int       *status)
{

/* Local variables */
   char type_str[DAT__SZTYP+1];

/* Enter routine   */

   if (*status != SAI__OK) return *status;

/* Construct the type string */
   datCctyp( len, type_str );

   datNew( locator, name_str, type_str, ndim, dims, status );

   return *status;
}

/*================================================*/
/* DAT_NEW1 - Create a vector structure component */
/*================================================*/

int
datNew1( const HDSLoc * locator,
	 const char * name,
	 const char * type,
	 size_t len,
	 int * status )
{
  hdsdim dims[1];

  if (*status != SAI__OK) return *status;

  dims[0] = (hdsdim)len;
  datNew( locator, name, type, 1, dims, status );
  return *status;
}

/*================================================*/
/* DAT_NEW1D - Create a vector double component */
/*================================================*/

int
datNew1D( const HDSLoc * locator,
	 const char * name,
	 size_t len,
	 int * status )
{
  if (*status != SAI__OK) return *status;
  datNew1( locator, name, "_DOUBLE", len, status );
  return *status;
}

/*================================================*/
/* DAT_NEW1I - Create a vector integer  component */
/*================================================*/

int
datNew1I( const HDSLoc * locator,
	 const char * name,
	 size_t len,
	 int * status )
{
  if (*status != SAI__OK) return *status;
  datNew1( locator, name, "_INTEGER", len, status );
  return *status;
}

/*========================================================*/
/* DAT_NEW1K - Create a vector 64-bit integer component */
/*========================================================*/

int
datNew1K( const HDSLoc * locator,
	 const char * name,
	 size_t len,
	 int * status )
{
  if (*status != SAI__OK) return *status;
  datNew1( locator, name, "_INT64", len, status );
  return *status;
}

/*======================================================*/
/* DAT_NEW1W - Create a vector short integer  component */
/*======================================================*/

int
datNew1W( const HDSLoc * locator,
	 const char * name,
	 size_t len,
	 int * status )
{
  if (*status != SAI__OK) return *status;
  datNew1( locator, name, "_WORD", len, status );
  return *status;
}

/*================================================================*/
/* DAT_NEW1UW - Create a vector unsigned short integer  component */
/*================================================================*/

int
datNew1UW( const HDSLoc * locator,
	 const char * name,
	 size_t len,
	 int * status )
{
  if (*status != SAI__OK) return *status;
  datNew1( locator, name, "_UWORD", len, status );
  return *status;
}

/*================================================*/
/* DAT_NEW1L - Create a vector logical component */
/*================================================*/

int
datNew1L( const HDSLoc * locator,
	 const char * name,
	 size_t len,
	 int * status )
{
  if (*status != SAI__OK) return *status;
  datNew1( locator, name, "_LOGICAL", len, status );
  return *status;
}

/*================================================*/
/* DAT_NEW1R - Create a vector real component */
/*================================================*/

int
datNew1R( const HDSLoc * locator,
	 const char * name,
	 size_t len,
	 int * status )
{
  if (*status != SAI__OK) return *status;
  datNew1( locator, name, "_REAL", len, status );
  return *status;
}


/*================================================*/
/* DAT_NEW1C - Create a vector CHAR component */
/*================================================*/

int
datNew1C( const HDSLoc * locator,
	  const char * name,
	  size_t len,
	  size_t nelem,
	  int * status )
{
  char type[DAT__SZTYP+1];

  if (*status != SAI__OK) return *status;
  datCctyp( len, type );
  datNew1( locator, name, type, nelem, status );
  return *status;
}

