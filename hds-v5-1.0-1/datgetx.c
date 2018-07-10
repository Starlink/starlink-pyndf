#if HAVE_CONFIG_H
#  include <config.h>
#endif

/*+DATGET.C-*/

/* Include files */
#include <stdio.h>                  /* stdio for sprintf() prototype         */

#include "ems.h"                    /* EMS error reporting routines          */

#include "hds1.h"                  /* Global definitions for HDS             */
#include "dat1.h"                  /* Internal dat_ definitions              */
#include "dat_err.h"               /* DAT__ error code definitions           */

#include "hds.h"
#include "sae_par.h"

/*==============================*/
/* DAT_GETI - Read Integer data */
/*==============================*/
int
datGetI(const HDSLoc    *locator,
        int       ndim,
        const hdsdim dims[],
        int       values[],
        int       *status)
{
#undef context_name
#undef context_message
#define context_name "DAT_GETI_ERR"
#define context_message\
        "DAT_GETI: Error reading integer value(s) from an HDS primitive."

   datGet(locator, "_INTEGER", ndim, dims,
                     values, status );

   return *status;
}

/*=======================================*/
/* DAT_GETK - Read 64-bit Integer data */
/*=======================================*/
int
datGetK(const HDSLoc    *locator,
        int       ndim,
        const hdsdim dims[],
        int64_t   values[],
        int       *status)
{
#undef context_name
#undef context_message
#define context_name "DAT_GETK_ERR"
#define context_message\
        "DAT_GETK: Error reading 64-bit integer value(s) from an HDS primitive."

   datGet(locator, "_INT64", ndim, dims,
                     values, status );

   return *status;
}

/*====================================*/
/* DAT_GETW - Read Short Integer data */
/*====================================*/
int
datGetW(const HDSLoc    *locator,
        int       ndim,
        const hdsdim dims[],
        short       values[],
        int       *status)
{
#undef context_name
#undef context_message
#define context_name "DAT_GETW_ERR"
#define context_message\
        "DAT_GETW: Error reading short integer value(s) from an HDS primitive."

   datGet(locator, "_WORD", ndim, dims,
                     values, status );

   return *status;
}

/*==============================================*/
/* DAT_GETUW - Read Unsigned Short Integer data */
/*==============================================*/
int
datGetUW(const HDSLoc    *locator,
        int       ndim,
        const hdsdim dims[],
        unsigned short  values[],
        int       *status)
{
#undef context_name
#undef context_message
#define context_name "DAT_GETUW_ERR"
#define context_message\
        "DAT_GETW: Error reading unsigned short integer value(s) from an HDS primitive."

   datGet(locator, "_UWORD", ndim, dims,
                     values, status );

   return *status;
}

/*===========================*/
/* DAT_GETR - Read REAL data */
/*===========================*/
int
datGetR(const HDSLoc    *locator,
        int       ndim,
        const hdsdim dims[],
        float     values[],
        int       *status)
{
#undef context_name
#undef context_message
#define context_name "DAT_GETR_ERR"
#define context_message\
        "DAT_GETR: Error reading real value(s) from an HDS primitive."

   datGet(locator, "_REAL", ndim, dims,
                      (unsigned char *) values, status );

   return *status;
}

/*=======================================*/
/* DAT_GETD - Read DOUBLE PRECISION data */
/*=======================================*/
int
datGetD(const HDSLoc    *locator,
        int       ndim,
        const hdsdim dims[],
        double    values[],
        int       *status)
{
#undef context_name
#undef context_message
#define context_name "DAT_GETD_ERR"
#define context_message\
    "DAT_GETD: Error reading double precision value(s) from an HDS primitive."

   datGet(locator, "_DOUBLE", ndim, dims,
                     values, status );

   return *status;
}

/*==============================*/
/* DAT_GETL - Read LOGICAL data */
/*==============================*/
int
datGetL(const HDSLoc    *locator,
        int       ndim,
        const hdsdim dims[],
        hdsbool_t       values[],
        int       *status)
{
#undef context_name
#undef context_message
#define context_name "DAT_GETL_ERR"
#define context_message\
        "DAT_GETL: Error reading logical value(s) from an HDS primitive."

   datGet(locator, "_LOGICAL", ndim, dims,
                     values, status );

   return *status;
}

/*================================*/
/* DAT_GETC - Read CHARACTER data */
/*================================*/
int
datGetC(const HDSLoc    *locator,
        int       ndim,
        const hdsdim dims[],
        char      values[],
        size_t    char_len,
        int       *status)
{
#undef context_name
#undef context_message
#define context_name "DAT_GETC_ERR"
#define context_message\
        "DAT_GETC: Error reading character value(s) from an HDS primitive."
   char stype[DAT__SZTYP+1];

   datCctyp( char_len, stype );
   datGet(locator, stype, ndim, dims, values, status );

   return *status;
}

/*==========================================================*/
/*                                                          */
/*                          GET 1x                          */
/*                                                          */
/*==========================================================*/

/*==================================*/
/* DAT_GET1D - Read 1D Double array */
/*==================================*/

int
datGet1D( const HDSLoc * locator,
	  size_t maxval,
	  double values[],
	  size_t *actval,
	  int * status ) {

  hdsdim dims[1];

  if (*status != SAI__OK) return *status;

  datSize( locator, actval, status );

  if ( *status == SAI__OK && maxval < *actval ) {
    *status = DAT__BOUND;
    emsSeti( "IN", (int)maxval );
    emsSeti( "SZ", (int)*actval );
    emsRep( "DAT_GET1D_ERR", "datGet1D: Bounds mismatch: ^IN < ^SZ", status);
  } else {
    dims[0] = *actval;
    datGetD( locator, 1, dims, values, status );
  }
  return *status;
}

/*==================================*/
/* DAT_GET1I - Read 1D Integer array */
/*==================================*/

int
datGet1I( const HDSLoc * locator,
	  size_t maxval,
	  int values[],
	  size_t *actval,
	  int * status ) {

  hdsdim dims[1];

  if (*status != SAI__OK) return *status;

  datSize( locator, actval, status );

  if ( *status == SAI__OK && maxval < *actval ) {
    *status = DAT__BOUND;
    emsSeti( "IN", (int)maxval );
    emsSeti( "SZ", (int)*actval );
    emsRep( "DAT_GET1I_ERR", "datGet1I: Bounds mismatch: ^IN < ^SZ", status);
  } else {
    dims[0] = *actval;
    datGetI( locator, 1, dims, values, status );
  }
  return *status;
}

/*==================================*/
/* DAT_GET1K - Read 1D 64-bit int array */
/*==================================*/

int
datGet1K( const HDSLoc * locator,
	  size_t maxval,
	  int64_t values[],
	  size_t *actval,
	  int * status ) {

  hdsdim dims[1];

  if (*status != SAI__OK) return *status;

  datSize( locator, actval, status );

  if ( *status == SAI__OK && maxval < *actval ) {
    *status = DAT__BOUND;
    emsSeti( "IN", (int)maxval );
    emsSeti( "SZ", (int)*actval );
    emsRep( "DAT_GET1I_ERR", "datGet1K: Bounds mismatch: ^IN < ^SZ", status);
  } else {
    dims[0] = *actval;
    datGetK( locator, 1, dims, values, status );
  }
  return *status;
}

/*==================================*/
/* DAT_GET1W - Read 1D 16-bit int array */
/*==================================*/

int
datGet1W( const HDSLoc * locator,
	  size_t maxval,
	  short values[],
	  size_t *actval,
	  int * status ) {

  hdsdim dims[1];

  if (*status != SAI__OK) return *status;

  datSize( locator, actval, status );

  if ( *status == SAI__OK && maxval < *actval ) {
    *status = DAT__BOUND;
    emsSeti( "IN", (int)maxval );
    emsSeti( "SZ", (int)*actval );
    emsRep( "DAT_GET1I_ERR", "datGetW: Bounds mismatch: ^IN < ^SZ", status);
  } else {
    dims[0] = *actval;
    datGetW( locator, 1, dims, values, status );
  }
  return *status;
}

/*==================================*/
/* DAT_GET1UW - Read 1D 16-bit unsigned int array */
/*==================================*/

int
datGet1UW( const HDSLoc * locator,
	  size_t maxval,
	  unsigned short values[],
	  size_t *actval,
	  int * status ) {

  hdsdim dims[1];

  if (*status != SAI__OK) return *status;

  datSize( locator, actval, status );

  if ( *status == SAI__OK && maxval < *actval ) {
    *status = DAT__BOUND;
    emsSeti( "IN", (int)maxval );
    emsSeti( "SZ", (int)*actval );
    emsRep( "DAT_GET1I_ERR", "datGetUW: Bounds mismatch: ^IN < ^SZ", status);
  } else {
    dims[0] = *actval;
    datGetUW( locator, 1, dims, values, status );
  }
  return *status;
}

/*==================================*/
/* DAT_GET1R - Read 1D float array */
/*==================================*/

int
datGet1R( const HDSLoc * locator,
	  size_t maxval,
	  float values[],
	  size_t *actval,
	  int * status ) {

  hdsdim dims[1];

  if (*status != SAI__OK) return *status;

  datSize( locator, actval, status );

  if ( *status == SAI__OK && maxval < *actval ) {
    *status = DAT__BOUND;
    emsSeti( "IN", (int)maxval );
    emsSeti( "SZ", (int)*actval );
    emsRep( "DAT_GET1R_ERR", "datGet1R: Bounds mismatch: ^IN < ^SZ", status);
  } else {
    dims[0] = *actval;
    datGetR( locator, 1, dims, values, status );
  }
  return *status;
}

/*==================================*/
/* DAT_GET1L - Read 1D Logical array */
/*==================================*/

int
datGet1L( const HDSLoc * locator,
	  size_t maxval,
	  hdsbool_t values[],
	  size_t *actval,
	  int * status ) {

  hdsdim dims[1];

  if (*status != SAI__OK) return *status;

  datSize( locator, actval, status );

  if ( *status == SAI__OK && maxval < *actval ) {
    *status = DAT__BOUND;
    emsSeti( "IN", (int)maxval );
    emsSeti( "SZ", (int)*actval );
    emsRep( "DAT_GET1L_ERR", "datGet1L: Bounds mismatch: ^IN < ^SZ", status);
  } else {
    dims[0] = *actval;
    datGetL( locator, 1, dims, values, status );
  }
  return *status;
}

/*==========================================================*/
/*                                                          */
/*                          GET Vx                          */
/*                                                          */
/*==========================================================*/

/*==========================================*/
/* DAT_GETVD - Read vectorized Double array */
/*==========================================*/

int
datGetVD( const HDSLoc * locator,
	  size_t maxval,
	  double values[],
	  size_t *actval,
	  int * status ) {
  HDSLoc * vec = NULL;
  if (*status != SAI__OK) return *status;
  datVec(locator, &vec, status );
  datGet1D( vec, maxval, values, actval, status );
  datAnnul( &vec, status );
  return *status;
}
/*==========================================*/
/* DAT_GETVI - Read vectorized Integer array */
/*==========================================*/

int
datGetVI( const HDSLoc * locator,
	  size_t maxval,
	  int values[],
	  size_t *actval,
	  int * status ) {
  HDSLoc * vec = NULL;
  if (*status != SAI__OK) return *status;
  datVec(locator, &vec, status );
  datGet1I( vec, maxval, values, actval, status );
  datAnnul( &vec, status );
  return *status;
}

/*==========================================*/
/* DAT_GETVK - Read vectorized Int64 array  */
/*==========================================*/

int
datGetVK( const HDSLoc * locator,
	    size_t maxval,
	    int64_t values[],
	    size_t *actval,
	    int * status ) {
  HDSLoc * vec = NULL;
  if (*status != SAI__OK) return *status;
  datVec(locator, &vec, status );
  datGet1K( vec, maxval, values, actval, status );
  datAnnul( &vec, status );
  return *status;
}

/*==========================================*/
/* DAT_GETVR - Read vectorized REAL array */
/*==========================================*/

int
datGetVR( const HDSLoc * locator,
	  size_t maxval,
	  float values[],
	  size_t *actval,
	  int * status ) {
  HDSLoc * vec = NULL;
  if (*status != SAI__OK) return *status;
  datVec(locator, &vec, status );
  datGet1R( vec, maxval, values, actval, status );
  datAnnul( &vec, status );
  return *status;
}

/*==========================================*/
/* DAT_GETVL - Read vectorized Logical array */
/*==========================================*/

int
datGetVL( const HDSLoc * locator,
	  size_t maxval,
	  hdsbool_t values[],
	  size_t *actval,
	  int * status ) {
  HDSLoc * vec = NULL;
  if (*status != SAI__OK) return *status;
  datVec(locator, &vec, status );
  datGet1L( vec, maxval, values, actval, status );
  datAnnul( &vec, status );
  return *status;
}
