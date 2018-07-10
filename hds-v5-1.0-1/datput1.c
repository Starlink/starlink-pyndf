
/*+DATPUT1.C-*/

/* Include files */

#include <string.h>
#include <stdio.h>                  /* stdio for sprintf() prototype         */

#include "ems.h"                    /* EMS error reporting routines          */

#include "hds1.h"                   /* Global definitions for HDS            */
#include "dat1.h"                   /* Internal dat_ definitions             */
#include "dat_err.h"                /* DAT__ error code definitions          */

#include "sae_par.h"
#include "hds.h"


/*===============================*/
/* DAT_PUTI - Write Integer data */
/*===============================*/
int
datPutI(const HDSLoc   *locator,
         int     ndim,
        const hdsdim dims[],
        const int     values[],
         int     *status)
{
#undef context_name
#undef context_message
#define context_name "DAT_PUTI_ERR"
#define context_message\
        "DAT_PUTI: Error writing integer values to an HDS primitive."

   datPut(locator,
          "_INTEGER",
          ndim,
          dims,
          values,
          status);
   return *status;
}

/*========================================*/
/* DAT_PUTK - Write 64-bit Integer data */
/*========================================*/
int
datPutK(const HDSLoc   *locator,
         int     ndim,
        const hdsdim dims[],
        const int64_t     values[],
         int     *status)
{
#undef context_name
#undef context_message
#define context_name "DAT_PUTK_ERR"
#define context_message\
        "DAT_PUTK: Error writing 64-bit integer values to an HDS primitive."

   datPut(locator,
          "_INT64",
          ndim,
          dims,
          values,
          status);
   return *status;
}

/*=====================================*/
/* DAT_PUTW - Write Short Integer data */
/*=====================================*/
int
datPutW(const HDSLoc   *locator,
        int     ndim,
        const hdsdim dims[],
        const short     values[],
        int     *status)
{
#undef context_name
#undef context_message
#define context_name "DAT_PUTW_ERR"
#define context_message\
        "DAT_PUTW: Error writing short integer values to an HDS primitive."

   datPut(locator,
          "_WORD",
          ndim,
          dims,
          values,
          status);
   return *status;
}

/*===============================================*/
/* DAT_PUTUW - Write Unsigned Short Integer data */
/*===============================================*/
int
datPutUW(const HDSLoc   *locator,
         int     ndim,
         const hdsdim dims[],
         const unsigned short values[],
         int     *status)
{
#undef context_name
#undef context_message
#define context_name "DAT_PUTUW_ERR"
#define context_message\
        "DAT_PUTUW: Error writing unsigned short integer values to an HDS primitive."

   datPut(locator,
          "_UWORD",
          ndim,
          dims,
          values,
          status);
   return *status;
}

/*============================*/
/* DAT_PUTR - Write Real data */
/*============================*/
int
datPutR( const HDSLoc    *locator,
         int       ndim,
         const hdsdim dims[],
         const float     values[],
         int       *status)
{
#undef context_name
#undef context_message
#define context_name "DAT_PUTR_ERR"
#define context_message\
        "DAT_PUTR: Error writing real values to an HDS primitive."

  datPut(locator,
          "_REAL",
          ndim,
          dims,
          values,
          status);
     return *status;
}

/*========================================*/
/* DAT_PUTD - Write Double precision data */
/*========================================*/
int
datPutD( const HDSLoc    *locator,
         int       ndim,
         const hdsdim dims[],
         const double    values[],
         int       *status)
{
#undef context_name
#undef context_message
#define context_name "DAT_PUTD_ERR"
#define context_message\
     "DAT_PUTD: Error writing double precision value(s) to an HDS primitive."

   datPut(locator,
          "_DOUBLE",
          ndim,
          dims,
          values,
          status);
     return *status;
}

/*===============================*/
/* DAT_PUTL - Write Logical data */
/*===============================*/
int
datPutL( const HDSLoc    *locator,
         int       ndim,
         const hdsdim dims[],
         const hdsbool_t values[],
         int       *status)
{
#undef context_name
#undef context_message
#define context_name "DAT_PUTL_ERR"
#define context_message\
        "DAT_PUTL: Error writing logical values to an HDS primitive."

     datPut(locator,
          "_LOGICAL",
          ndim,
          dims,
          values,
          status);
     return *status;
}

/*=================================*/
/* DAT_PUTC - Write Character data */
/*=================================*/
int
datPutC( const HDSLoc    *locator,
         int       ndim,
         const hdsdim dims[],
         const char      string[],
         size_t    string_length,
         int       *status)
{
/* Local variables */
  char *string1;
  char stype[DAT__SZTYP+1];

#undef context_name
#undef context_message
#define context_name "DAT_PUTC_ERR"
#define context_message\
        "DAT_PUTC: Error writing character value(s) to an HDS primitive."

/* Encode the (fixed) string length into the primitive type definition  */
/* before calling datPut                                                */
/* Assume that a zero length string (eg. prefix=\"\") is a single space */
/* to make consistent with earlier HDS behaviour!                       */
   if( string_length > 0 ) {
      datCctyp( string_length, stype );
      string1 = (char*)string;
   } else {
      strcpy( stype, "_CHAR" );
      string1 = " ";
   }
   datPut(locator,
          stype,
          ndim,
          dims,
          string1,
          status);
     return *status;
}

/*         O N E - D I M           P U T     */

/*=================================*/
/* DAT_PUT1D - Write 1D double array */
/*=================================*/

int
datPut1D( const HDSLoc * locator,
	  size_t nval,
	  const double values[],
	  int * status ) {
  size_t size;
  hdsdim dim[1];

  if ( *status != SAI__OK ) return *status;
  datSize( locator, &size, status );
  if ( *status == SAI__OK && size != nval ) {
    *status = DAT__BOUND;
    emsSeti( "IN", (int)nval );
    emsSeti( "SZ", (int)size );
    emsRep( "DAT_PUT1D_ERR", "Bounds mismatch: ^IN != ^SZ", status);
  } else {
    dim[0] = (hdsdim)size;
    datPutD( locator, 1, dim, values, status );
  }
  return *status;
}

/*=================================*/
/* DAT_PUT1I - Write 1D int array */
/*=================================*/

int
datPut1I( const HDSLoc * locator,
	  size_t nval,
	  const int values[],
	  int * status ) {
  size_t size;
  hdsdim dim[1];

  if ( *status != SAI__OK ) return *status;
  datSize( locator, &size, status );
  if ( *status == SAI__OK && size != nval ) {
    *status = DAT__BOUND;
    emsSeti( "IN", (int)nval );
    emsSeti( "SZ", (int)size );
    emsRep( "DAT_PUT1I_ERR", "Bounds mismatch: ^IN != ^SZ", status);
  } else {
    dim[0] = (hdsdim)size;
    datPutI( locator, 1, dim, values, status );
  }
  return *status;
}

/*=================================*/
/* DAT_PUT1K - Write 1D 64-bit int array */
/*=================================*/

int
datPut1K( const HDSLoc * locator,
	  size_t nval,
	  const int64_t values[],
	  int * status ) {
  size_t size;
  hdsdim dim[1];

  if ( *status != SAI__OK ) return *status;
  datSize( locator, &size, status );
  if ( *status == SAI__OK && size != nval ) {
    *status = DAT__BOUND;
    emsSeti( "IN", (int)nval );
    emsSeti( "SZ", (int)size );
    emsRep( "DAT_PUT1K_ERR", "Bounds mismatch: ^IN != ^SZ", status);
  } else {
    dim[0] = (hdsdim)size;
    datPutK( locator, 1, dim, values, status );
  }
  return *status;
}

/*======================================*/
/* DAT_PUT1W - Write 1D short int array */
/*======================================*/

int
datPut1W( const HDSLoc * locator,
	  size_t nval,
	  const short values[],
	  int * status ) {
  size_t size;
  hdsdim dim[1];

  if ( *status != SAI__OK ) return *status;
  datSize( locator, &size, status );
  if ( *status == SAI__OK && size != nval ) {
    *status = DAT__BOUND;
    emsSeti( "IN", (int)nval );
    emsSeti( "SZ", (int)size );
    emsRep( "DAT_PUT1W_ERR", "Bounds mismatch: ^IN != ^SZ", status);
  } else {
    dim[0] = (hdsdim)size;
    datPutW( locator, 1, dim, values, status );
  }
  return *status;
}

/*================================================*/
/* DAT_PUT1UW - Write 1D unsigned short int array */
/*================================================*/

int
datPut1UW( const HDSLoc * locator,
	   size_t nval,
	   const unsigned short values[],
	   int * status ) {
  size_t size;
  hdsdim dim[1];

  if ( *status != SAI__OK ) return *status;
  datSize( locator, &size, status );
  if ( *status == SAI__OK && size != nval ) {
    *status = DAT__BOUND;
    emsSeti( "IN", (int)nval );
    emsSeti( "SZ", (int)size );
    emsRep( "DAT_PUT1UW_ERR", "Bounds mismatch: ^IN != ^SZ", status);
  } else {
    dim[0] = (hdsdim)size;
    datPutUW( locator, 1, dim, values, status );
  }
  return *status;
}

/*=================================*/
/* DAT_PUT1R - Write 1D float array */
/*=================================*/

int
datPut1R( const HDSLoc * locator,
	  size_t nval,
	  const float values[],
	  int * status ) {
  size_t size;
  hdsdim dim[1];

  if ( *status != SAI__OK ) return *status;
  datSize( locator, &size, status );
  if ( *status == SAI__OK && size != nval ) {
    *status = DAT__BOUND;
    emsSeti( "IN", (int)nval );
    emsSeti( "SZ", (int)size );
    emsRep( "DAT_PUT1R_ERR", "Bounds mismatch: ^IN != ^SZ", status);
  } else {
    dim[0] = (hdsdim)size;
    datPutR( locator, 1, dim, values, status );
  }
  return *status;
}

/*=================================*/
/* DAT_PUT1L - Write 1D logical array */
/*=================================*/

int
datPut1L( const HDSLoc * locator,
	  size_t nval,
	  const hdsbool_t values[],
	  int * status ) {
  size_t size;
  hdsdim dim[1];

  if ( *status != SAI__OK ) return *status;
  datSize( locator, &size, status );
  if ( *status == SAI__OK && size != nval ) {
    *status = DAT__BOUND;
    emsSeti( "IN", (int)nval );
    emsSeti( "SZ", (int)size );
    emsRep( "DAT_PUT1L_ERR", "Bounds mismatch: ^IN != ^SZ", status);
  } else {
    dim[0] = (hdsdim)size;
    datPutL( locator, 1, dim, values, status );
  }
  return *status;
}

/*         V E C T O R I Z E D     P U T     */

/*=================================*/
/* DAT_PUTVD - Write vectorized doubles */
/*=================================*/

int
datPutVD( const HDSLoc * locator,
	  size_t nval,
	  const double values[],
	  int *status ) {
  HDSLoc *vec = NULL;
  datVec( locator, &vec, status );
  datPut1D( vec, nval, values, status );
  datAnnul( &vec, status );
  return *status;
}

/*==================================*/
/* DAT_PUTVI - Write vectorized int */
/*==================================*/

int
datPutVI( const HDSLoc * locator,
	  size_t nval,
	  const int values[],
	  int *status ) {
  HDSLoc *vec = NULL;
  datVec( locator, &vec, status );
  datPut1I( vec, nval, values, status );
  datAnnul( &vec, status );
  return *status;
}

/*====================================*/
/* DAT_PUTVK - Write vectorized int64 */
/*====================================*/

int
datPutVK( const HDSLoc * locator,
	    size_t nval,
	    const int64_t values[],
	    int *status ) {
  HDSLoc *vec = NULL;
  datVec( locator, &vec, status );
  datPut1K( vec, nval, values, status );
  datAnnul( &vec, status );
  return *status;
}

/*====================================*/
/* DAT_PUTVR - Write vectorized float */
/*====================================*/

int
datPutVR( const HDSLoc * locator,
	  size_t nval,
	  const float values[],
	  int *status ) {
  HDSLoc *vec = NULL;
  datVec( locator, &vec, status );
  datPut1R( vec, nval, values, status );
  datAnnul( &vec, status );
  return *status;
}

/*=================================*/
/* DAT_PUTVL - Write vectorized logical */
/*=================================*/

int
datPutVL( const HDSLoc * locator,
	  size_t nval,
	  const hdsbool_t values[],
	  int *status ) {
  HDSLoc *vec = NULL;
  datVec( locator, &vec, status );
  datPut1L( vec, nval, values, status );
  datAnnul( &vec, status );
  return *status;
}
