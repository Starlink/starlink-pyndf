/* Protect against multiple inclusion */
#ifndef STAR_HDS_V5_H_INCLUDED
#define STAR_HDS_V5_H_INCLUDED

#include "dat_par.h"

/* Relative location of type definitions depends on whether we are
building the library or using the installed version */
#if HDS_INTERNAL_INCLUDES
#  include "hds_types.h"
#else
#  include "star/hds_types.h"
#endif

/*=================================*/
/* datAlter - Alter size of object */
/*=================================*/

int
datAlter_v5(HDSLoc *locator, int ndim, const hdsdim dims[], int *status);

/*==========================*/
/* datAnnul - Annul locator */
/*==========================*/

int
datAnnul_v5(HDSLoc **locator, int *status);

/*==============================================*/
/* datBasic - Map data (in basic machine units) */
/*==============================================*/

int
datBasic_v5(const HDSLoc *locator, const char *mode_c, unsigned char **pntr, size_t *len, int *status);

/*=====================================*/
/* datCcopy - copy one structure level */
/*=====================================*/

int
datCcopy_v5(const HDSLoc *locator1, const HDSLoc *locator2, const char *name, HDSLoc **locator3, int *status);

/*=======================================*/
/* datCctyp - construct _CHAR*nnn string */
/*=======================================*/

void
datCctyp_v5(size_t size, char type[DAT__SZTYP+1]);


/*===========================================*/
/* datCell - Locate a "cell" (array element) */
/*===========================================*/

int
datCell_v5(const HDSLoc *locator1, int ndim, const hdsdim subs[], HDSLoc **locator2, int *status);

/*=================================================*/
/* datChscn - validate the supplied component name */
/*=================================================*/

int
datChscn_v5(const char * name, int *status);

/*==========================================*/
/* datClen - Obtain character string length */
/*==========================================*/

int
datClen_v5(const HDSLoc *locator, size_t *clen, int *status);

/*===========================*/
/* datClone - clone locator */
/*===========================*/

int
datClone_v5(const HDSLoc *locator1, HDSLoc **locator2, int *status);

/*================================*/
/* datCoerc - coerce object shape */
/*================================*/

int
datCoerc_v5(const HDSLoc *locator1, int ndim, HDSLoc **locator2, int *status);

/*=======================*/
/* datCopy - copy object */
/*=======================*/

int
datCopy_v5(const HDSLoc *locator1, const HDSLoc *locator2, const char *name_c, int *status);

/*============================================================*/
/* datDrep - Obtain primitive data representation information */
/*============================================================*/

int
datDrep_v5(const HDSLoc *locator, char **format_str, char **order_str, int *status);

/*========================================*/
/* datErase - Erase object                */
/*========================================*/

int
datErase_v5(const HDSLoc *locator, const char *name_str, int *status);

/*===========================================================*/
/* datErmsg - Translate a status value into an error message */
/*===========================================================*/

int
datErmsg_v5(int status, size_t *len, char *msg_str);

/*================================*/
/* datFind - Find named component */
/*================================*/

int
datFind_v5(const HDSLoc *locator1, const char *name_str, HDSLoc **locator2, int *status);

/*============================*/
/* datGet - Read primitive(s) */
/*============================*/

int
datGet_v5(const HDSLoc *locator, const char *type_str, int ndim, const hdsdim dims[], void *values, int *status);

/*===================================*/
/* datGetC - Read _CHAR primitive(s) */
/*===================================*/

int
datGetC_v5(const HDSLoc *locator, const int ndim, const hdsdim dims[], char values[], size_t char_len, int *status);

/*=====================================*/
/* datGetD - Read _DOUBLE primitive(s) */
/*=====================================*/

int
datGetD_v5(const HDSLoc *locator, int ndim, const hdsdim dims[], double values[], int *status);

/*======================================*/
/* datGetI - Read _INTEGER primitive(s) */
/*======================================*/

int
datGetI_v5(const HDSLoc *locator, int ndim, const hdsdim dims[], int values[], int *status);

/*======================================*/
/* datGetK - Read _INT64 primitive(s) */
/*======================================*/

int
datGetK_v5(const HDSLoc *locator, int ndim, const hdsdim dims[], int64_t values[], int *status);

/*===================================*/
/* datGetW - Read _WORD primitive(s) */
/*===================================*/

int
datGetW_v5(const HDSLoc *locator, int ndim, const hdsdim dims[], short values[], int *status);

/*===================================*/
/* datGetUW - Read _UWORD primitive(s) */
/*===================================*/

int
datGetUW_v5(const HDSLoc *locator, int ndim, const hdsdim dims[], unsigned short values[], int *status);

/*======================================*/
/* datGetL - Read _LOGICAL primitive(s) */
/*======================================*/

int
datGetL_v5(const HDSLoc *locator, int ndim, const hdsdim dims[], hdsbool_t values[], int *status);

/*===================================*/
/* datGetR - Read _REAL primitive(s) */
/*===================================*/

int
datGetR_v5(const HDSLoc *locator, int ndim, const hdsdim dims[], float values[], int *status);

/*======================================*/
/* datGet0C - Read scalar string value  */
/*======================================*/

int
datGet0C_v5(const HDSLoc * locator, char * value, size_t len, int * status);

/*======================================*/
/* datGet0D - Read scalar double value  */
/*======================================*/

int
datGet0D_v5(const HDSLoc * locator, double * value, int * status);

/*=====================================*/
/* datGet0R - Read scalar float value  */
/*=====================================*/

int
datGet0R_v5(const HDSLoc * locator, float * value, int * status);

/*=======================================*/
/* datGet0I - Read scalar integer value  */
/*=======================================*/

int
datGet0I_v5(const HDSLoc * locator, int * value, int * status);

/*================================================*/
/* datGet0K - Read scalar 64-bit integer value  */
/*================================================*/

int
datGet0K_v5(const HDSLoc * locator, int64_t * value, int * status);

/*=============================================*/
/* datGet0W - Read scalar short integer value  */
/*=============================================*/

int
datGet0W_v5(const HDSLoc * locator, short * value, int * status);

/*=============================================*/
/* datGet0UW - Read scalar unsigned short integer value  */
/*=============================================*/

int
datGet0UW_v5(const HDSLoc * locator, unsigned short * value, int * status);

/*=======================================*/
/* datGet0L - Read scalar logical value  */
/*=======================================*/

int
datGet0L_v5(const HDSLoc * locator, hdsbool_t * value, int * status);

/*==================================*/
/* DAT_GET1C - Read 1D string array */
/*==================================*/

int
datGet1C_v5(const HDSLoc * locator, size_t maxval, size_t bufsize, char *buffer, char *pntrs[], size_t * actval, int * status);

/*==================================*/
/* DAT_GET1D - Read 1D Double array */
/*==================================*/

int
datGet1D_v5(const HDSLoc * locator, size_t maxval, double values[], size_t *actval, int * status);

/*==================================*/
/* DAT_GET1I - Read 1D Integer array */
/*==================================*/

int
datGet1I_v5(const HDSLoc * locator, size_t maxval, int values[], size_t *actval, int * status);

/*============================================*/
/* DAT_GET1K - Read 1D 64-bit Integer array */
/*============================================*/

int
datGet1K_v5(const HDSLoc * locator, size_t maxval, int64_t values[], size_t *actval, int * status);

/*=========================================*/
/* DAT_GET1W - Read 1D Short Integer array */
/*=========================================*/

int
datGet1W_v5(const HDSLoc * locator, size_t maxval, short values[], size_t *actval, int * status);

/*===================================================*/
/* DAT_GET1UW - Read 1D Unsigned Short Integer array */
/*===================================================*/

int
datGet1UW_v5(const HDSLoc * locator, size_t maxval, unsigned short values[], size_t *actval, int * status);

/*==================================*/
/* DAT_GET1R - Read 1D REAL array */
/*==================================*/

int
datGet1R_v5(const HDSLoc * locator, size_t maxval, float values[], size_t *actval, int * status);

/*==================================*/
/* DAT_GET1L - Read 1D Logical array */
/*==================================*/

int
datGet1L_v5(const HDSLoc * locator, size_t maxval, hdsbool_t values[], size_t *actval, int * status);

/*==================================*/
/* DAT_GETVC - Read vectorized 1D string array */
/*==================================*/

int
datGetVC_v5(const HDSLoc * locator, size_t maxval, size_t bufsize, char *buffer, char *pntrs[], size_t * actval, int * status);


/*==========================================*/
/* DAT_GETVD - Read vectorized Double array */
/*==========================================*/

int
datGetVD_v5(const HDSLoc * locator, size_t maxval, double values[], size_t *actval, int * status);

/*==========================================*/
/* DAT_GETVI - Read vectorized Integer array */
/*==========================================*/

int
datGetVI_v5(const HDSLoc * locator, size_t maxval, int values[], size_t *actval, int * status);

/*==========================================*/
/* DAT_GETVK - Read vectorized Int64 array */
/*==========================================*/

int
datGetVK_v5(const HDSLoc * locator, size_t maxval, int64_t values[], size_t *actval, int * status);

/*==========================================*/
/* DAT_GETVR - Read vectorized REAL array */
/*==========================================*/

int
datGetVR_v5(const HDSLoc * locator, size_t maxval, float values[], size_t *actval, int * status);

/*==========================================*/
/* DAT_GETVL - Read vectorized Logical array */
/*==========================================*/

int
datGetVL_v5(const HDSLoc * locator, size_t maxval, hdsbool_t values[], size_t *actval, int * status);


/*======================================*/
/* datIndex - Index into component list */
/*======================================*/

int
datIndex_v5(const HDSLoc *locator1, int index, HDSLoc **locator2, int *status);

/*===================================*/
/* datLen - Inquire primitive length */
/*===================================*/

int
datLen_v5(const HDSLoc *locator, size_t *len, int *status);

/*=========================================================*/
/* datLock - Lock an object for use by the current thread. */
/*=========================================================*/

int
datLock_v5( HDSLoc *locator, int recurs, int readonly, int *status);


/*=======================================================================*/
/* datLocked - See of an object is locked for use by the current thread. */
/*=======================================================================*/

int
datLocked_v5( const HDSLoc *locator, int recursive, int *status);

/*===========================*/
/* datMap - Map primitive(s) */
/*===========================*/

int
datMap_v5(HDSLoc *locator, const char *type_str, const char *mode_str, int ndim, const hdsdim dims[], void **pntr, int *status);

/*==================================*/
/* datMapC - Map _CHAR primitive(s) */
/*==================================*/

int
datMapC_v5(HDSLoc *locator, const char *mode_str, int ndim, const hdsdim dims[], unsigned char **pntr, int *status);

/*====================================*/
/* datMapD - Map _DOUBLE primitive(s) */
/*====================================*/

int
datMapD_v5(HDSLoc *locator, const char *mode_str, int ndim, const hdsdim dims[], double **pntr, int *status);

/*=====================================*/
/* datMapI - Map _INTEGER primitive(s) */
/*=====================================*/

int
datMapI_v5(HDSLoc *locator, const char *mode_str, int ndim, const hdsdim dims[], int **pntr, int *status);

/*=====================================*/
/* datMapK - Map _INT64 primitive(s) */
/*=====================================*/

int
datMapK_v5(HDSLoc *locator, const char *mode_str, int ndim, const hdsdim dims[], int **pntr, int *status);

/*=====================================*/
/* datMapL - Map _LOGICAL primitive(s) */
/*=====================================*/

int
datMapL_v5(HDSLoc *locator, const char *mode_str, int ndim, const hdsdim dims[], hdsbool_t **pntr, int *status);

/*==================================*/
/* datMapR - Map _REAL primitive(s) */
/*==================================*/

int
datMapR_v5(HDSLoc *locator, const char *mode_str, int ndim, const hdsdim dims[], float **pntr, int *status);


/*========================================*/
/* datMapN - Map primitive as N-dim array */
/*========================================*/

int
datMapN_v5(HDSLoc *locator, const char *type_str, const char *mode_str, int ndim, void **pntr, hdsdim dims[], int *status);

/*==================================*/
/* datMapV - Map vectorized primitive(s) */
/*==================================*/

int
datMapV_v5(HDSLoc *locator, const char *type_str, const char *mode_str, void **pntr, size_t *actval, int *status);


/*==================================*/
/* datMould - Alter shape of object */
/*==================================*/

int
datMould_v5(HDSLoc *locator, int ndim, const hdsdim dims[], int *status);

/*=======================*/
/* datMove - Move object */
/*=======================*/

int
datMove_v5(HDSLoc **locator1, const HDSLoc *locator2, const char *name_str, int *status);

/*======================================*/
/* datMsg - store filename in EMS token */
/*======================================*/

void
datMsg_v5(const char * token, const HDSLoc * locator);

/*===============================*/
/* datName - Enquire object name */
/*===============================*/

int
datName_v5(const HDSLoc *locator, char name_str[DAT__SZNAM+1], int *status);

/*=========================================*/
/* datNcomp - Inquire number of components */
/*=========================================*/

int
datNcomp_v5(const HDSLoc *locator, int *ncomp, int *status);

/*===============================*/
/* datNew - Create new component */
/*===============================*/

int
datNew_v5(const HDSLoc *locator, const char *name_str, const char *type_str, int ndim, const hdsdim dims[], int *status);

/*============================================*/
/* datNewC - Create new _CHAR type component */
/*============================================*/

int
datNewC_v5(const HDSLoc *locator, const char *name_str, size_t len, int ndim, const hdsdim dims[], int *status);

/*=======================================*/
/* datNew0 - Create new scalar component */
/*=======================================*/

int
datNew0_v5(const HDSLoc *locator, const char *name_str, const char *type_str, int *status);

/*===============================================*/
/* datNew0D - Create new scalar double component */
/*===============================================*/

int
datNew0D_v5(const HDSLoc *locator, const char *name_str, int *status);

/*================================================*/
/* datNew0I - Create new scalar integer component */
/*================================================*/

int
datNew0I_v5(const HDSLoc *locator, const char *name_str, int *status);

/*=========================================================*/
/* datNew0K - Create new scalar 64-bit integer component */
/*=========================================================*/

int
datNew0K_v5(const HDSLoc *locator, const char *name_str, int *status);

/*======================================================*/
/* datNew0W - Create new scalar short integer component */
/*======================================================*/

int
datNew0W_v5(const HDSLoc *locator, const char *name_str, int *status);

/*================================================================*/
/* datNew0UW - Create new scalar unsigned short integer component */
/*================================================================*/

int
datNew0UW_v5(const HDSLoc *locator, const char *name_str, int *status);

/*=============================================*/
/* datNew0R - Create new scalar real component */
/*=============================================*/

int
datNew0R_v5(const HDSLoc *locator, const char *name_str, int *status);

/*================================================*/
/* datNew0L - Create new scalar logical component */
/*================================================*/

int
datNew0L_v5(const HDSLoc *locator, const char *name_str, int *status);

/*================================================*/
/* datNew0L - Create new scalar logical component */
/*================================================*/

int
datNew0C_v5(const HDSLoc *locator, const char *name_str, size_t len, int *status);



/*=======================================*/
/* datNew1 - Create new vector component */
/*=======================================*/

int
datNew1_v5(const HDSLoc *locator, const char *name_str, const char *type_str, size_t len, int *status);

/*=======================================*/
/* datNew1C - Create new vector string  */
/*=======================================*/

int
datNew1C_v5(const HDSLoc *locator, const char *name_str, size_t len, size_t nelem, int *status);

/*=======================================*/
/* datNew1d - Create new vector double   */
/*=======================================*/

int
datNew1D_v5(const HDSLoc *locator, const char *name_str, size_t len, int *status);

/*=======================================*/
/* datNew1I - Create new vector integer  */
/*=======================================*/

int
datNew1I_v5(const HDSLoc *locator, const char *name_str, size_t len, int *status);

/*================================================*/
/* datNew1K - Create new vector 64-bit integer  */
/*================================================*/

int
datNew1K_v5(const HDSLoc *locator, const char *name_str, size_t len, int *status);

/*=============================================*/
/* datNew1W - Create new vector short integer  */
/*=============================================*/

int
datNew1W_v5(const HDSLoc *locator, const char *name_str, size_t len, int *status);

/*=======================================================*/
/* datNew1UW - Create new vector unsigned short integer  */
/*=======================================================*/

int
datNew1UW_v5(const HDSLoc *locator, const char *name_str, size_t len, int *status);

/*=======================================*/
/* datNew1L - Create new vector logical   */
/*=======================================*/

int
datNew1L_v5(const HDSLoc *locator, const char *name_str, size_t len, int *status);

/*=======================================*/
/* datNew1R - Create new vector float   */
/*=======================================*/

int
datNew1R_v5(const HDSLoc *locator, const char *name_str, size_t len, int *status);

/*====================================*/
/* datParen - Locate parent structure */
/*====================================*/

int
datParen_v5(const HDSLoc *locator1, HDSLoc **locator2, int *status);


/*=====================================*/
/* datPrec - Enquire storage precision */
/*=====================================*/

int
datPrec_v5(const HDSLoc *locator, size_t *nbytes, int *status);

/*====================================*/
/* datPrim - Enquire object primitive */
/*====================================*/

int
datPrim_v5(const HDSLoc *locator, hdsbool_t *prim, int *status);

/*=========================================================*/
/* datPrmry - Set/Enquire primary/secondary locator status */
/*=========================================================*/

int
datPrmry_v5(hdsbool_t set, HDSLoc **locator, hdsbool_t *prmry, int *status);

/*==================================*/
/* datPutC - Write _CHAR primitive */
/*==================================*/

int
datPutC_v5(const HDSLoc *locator, int ndim, const hdsdim dims[], const char string[], size_t string_length, int *status);

/*====================================*/
/* datPutD - Write _DOUBLE primitives */
/*====================================*/

int
datPutD_v5(const HDSLoc *locator, int ndim, const hdsdim dims[], const double values[], int *status);

/*=====================================*/
/* datPutI - Write _INTEGER primitives */
/*=====================================*/

int
datPutI_v5(const HDSLoc *locator, int ndim, const hdsdim dims[], const int values[], int *status);

/*=====================================*/
/* datPutK - Write _INT64 primitives */
/*=====================================*/

int
datPutK_v5(const HDSLoc *locator, int ndim, const hdsdim dims[], const int64_t values[], int *status);

/*=====================================*/
/* datPutW - Write _WORD primitives */
/*=====================================*/

int
datPutW_v5(const HDSLoc *locator, int ndim, const hdsdim dims[], const short values[], int *status);

/*====================================*/
/* datPutUW - Write _UWORD primitives */
/*====================================*/

int
datPutUW_v5(const HDSLoc *locator, int ndim, const hdsdim dims[], const unsigned short values[], int *status);

/*==================================*/
/* datPutR - Write _REAL primitives */
/*==================================*/

int
datPutR_v5(const HDSLoc *locator, int ndim, const hdsdim dims[], const float values[], int *status);

/*=====================================*/
/* datPutL - Write _LOGICAL primitives */
/*=====================================*/

int
datPutL_v5(const HDSLoc *locator, int ndim, const hdsdim dims[], const hdsbool_t values[], int *status);

/*==========================*/
/* datPut - Write primitive */
/*==========================*/

int
datPut_v5(const HDSLoc *locator, const char *type_str, int ndim, const hdsdim dims[], const void *values, int *status);

/*=======================================*/
/* datPut0C - Write scalar string value  */
/*=======================================*/

int
datPut0C_v5(const HDSLoc * locator, const char * value, int * status);

/*=======================================*/
/* datPut0D - Write scalar double value  */
/*=======================================*/

int
datPut0D_v5(const HDSLoc * locator, double value, int * status);

/*======================================*/
/* datPut0R - Write scalar float value  */
/*======================================*/

int
datPut0R_v5(const HDSLoc * locator, float value, int * status);

/*========================================*/
/* datPut0I - Write scalar integer value  */
/*========================================*/

int
datPut0I_v5(const HDSLoc * locator, int value, int * status);

/*========================================*/
/* datPut0I - Write scalar 64-bit integer value  */
/*========================================*/

int
datPut0K_v5(const HDSLoc * locator, int64_t value, int * status);

/*==============================================*/
/* datPut0W - Write scalar short integer value  */
/*===============================================*/

int
datPut0W_v5(const HDSLoc * locator, short value, int * status);

/*========================================================*/
/* datPut0UW - Write scalar unsigned short integer value  */
/*========================================================*/

int
datPut0UW_v5(const HDSLoc * locator, unsigned short value, int * status);

/*========================================*/
/* datPut0L - Write scalar logical value  */
/*========================================*/

int
datPut0L_v5(const HDSLoc * locator, hdsbool_t value, int * status);

/*========================================*/
/* datPut1C - Write 1D character array       */
/*========================================*/

int
datPut1C_v5(const HDSLoc * locator, size_t nval, const char *values[], int * status);

/*========================================*/
/* datPut1D - Write 1D double array       */
/*========================================*/

int
datPut1D_v5(const HDSLoc * locator, size_t nval, const double values[], int * status);

/*========================================*/
/* datPut1I - Write 1D int array       */
/*========================================*/

int
datPut1I_v5(const HDSLoc * locator, size_t nval, const int values[], int * status);

/*========================================*/
/* datPut1K - Write 1D 64-bit int array */
/*========================================*/

int
datPut1K_v5(const HDSLoc * locator, size_t nval, const int64_t values[], int * status);

/*===========================================*/
/* datPut1W - Write 1D short int array       */
/*===========================================*/

int
datPut1W_v5(const HDSLoc * locator, size_t nval, const short values[], int * status);

/*===============================================*/
/* datPut1UW - Write 1D unsigned short int array */
/*===============================================*/

int
datPut1UW_v5(const HDSLoc * locator, size_t nval, const unsigned short values[], int * status);

/*========================================*/
/* datPut1R - Write 1D double array       */
/*========================================*/

int
datPut1R_v5(const HDSLoc * locator, size_t nval, const float values[], int * status);

/*========================================*/
/* datPut1L - Write 1D Logical/int array       */
/*========================================*/

int
datPut1L_v5(const HDSLoc * locator, size_t nval, const hdsbool_t values[], int * status);

/*================================================*/
/* datPutVD - Write vectorized double array       */
/*================================================*/

int
datPutVD_v5(const HDSLoc * locator, size_t nval, const double values[], int * status);

/*================================================*/
/* datPutVI - Write vectorized int array       */
/*================================================*/

int
datPutVI_v5(const HDSLoc * locator, size_t nval, const int values[], int * status);

/*================================================*/
/* datPutVI - Write vectorized int64 array       */
/*================================================*/

int
datPutVK_v5(const HDSLoc * locator, size_t nval, const int64_t values[], int * status);

/*================================================*/
/* datPutVR - Write vectorized REAL/float array       */
/*================================================*/

int
datPutVR_v5(const HDSLoc * locator, size_t nval, const float values[], int * status);

/*================================================*/
/* datPutVL - Write vectorized Logical array       */
/*================================================*/

int
datPutVL_v5(const HDSLoc * locator, size_t nval, const hdsbool_t values[], int * status);

/*================================================*/
/* datPutVC - Write vectorized character array       */
/*================================================*/

int
datPutVC_v5(const HDSLoc * locator, size_t nval, const char *values[], int * status);


/*========================================*/
/* datRef - Enquire object reference name */
/*========================================*/

int
datRef_v5(const HDSLoc * locator, char * ref, size_t reflen, int *status);

/*===================================================*/
/* datRefct - Enquire container file reference count */
/*===================================================*/

int
datRefct_v5(const HDSLoc *locator, int *refct, int *status);

/*=============================*/
/* datRenam - Rename an object */
/*=============================*/

int
datRenam_v5(HDSLoc *locator, const char *name_str, int *status);

/*================================*/
/* datReset - Reset object state */
/*================================*/

int
datReset_v5(const HDSLoc *locator, int *status);

/*================================*/
/* datRetyp - Change object type */
/*================================*/

int
datRetyp_v5(const HDSLoc *locator, const char *type_str, int *status);

/*=================================*/
/* datShape - Enquire object shape */
/*=================================*/

int
datShape_v5(const HDSLoc *locator, int maxdim, hdsdim dims[], int *actdim, int *status);

/*===============================*/
/* datSize - Enquire object size */
/*===============================*/

int
datSize_v5(const HDSLoc *locator, size_t *size, int *status);

/*================================*/
/* datSlice - Locate object slice */
/*================================*/

int
datSlice_v5(const HDSLoc *locator1, int ndim, const hdsdim lower[], const hdsdim upper[], HDSLoc **locator2, int *status);

/*=================================*/
/* datState - Enquire object state */
/*=================================*/

int
datState_v5(const HDSLoc *locator, hdsbool_t *state, int *status);

/*=====================================*/
/* datStruc - Enquire object structure */
/*=====================================*/

int
datStruc_v5(const HDSLoc *locator, hdsbool_t *struc, int *status);

/*===================================*/
/* datTemp - Create temporary object */
/*===================================*/

int
datTemp_v5(const char *type_str, int ndim, const hdsdim dims[], HDSLoc **locator, int *status);

/*=========================================*/
/* datThere - Enquire component existence */
/*=========================================*/

int
datThere_v5(const HDSLoc *locator, const char *name_c, hdsbool_t *there, int *status);

/*===============================*/
/* datType - Enquire object type */
/*===============================*/

int
datType_v5(const HDSLoc *locator, char type_str[DAT__SZTYP + 1], int *status);

/*=============================================================*/
/* datNolock - Prevent lock echks being performed on an object */
/*=============================================================*/

int
datNolock_v5( HDSLoc *locator, int *status);

/*=============================================================*/
/* datUnlock - Unlock an object so another thread can lock it. */
/*=============================================================*/

int
datUnlock_v5( HDSLoc *locator, int recurs, int *status);

/*=========================*/
/* datUnmap - Unmap object */
/*=========================*/

int
datUnmap_v5(HDSLoc *locator, int *status);

/*==================================*/
/* datValid - Enquire locator valid */
/*==================================*/

int
datValid_v5(const HDSLoc *locator, hdsbool_t *valid, int *status);

/*===========================*/
/* datVec - Vectorise object */
/*===========================*/

int
datVec_v5(const HDSLoc *locator1, HDSLoc **locator2, int *status);

/*================================================*/
/* datWhere - Find primitive position in HDS file */
/*            Currently not part of the public    */
/*            C API                               */
/*================================================*/

/*==================================================*/
/* hdsCopy - Copy an object to a new container file */
/*==================================================*/

int
hdsCopy_v5(const HDSLoc *locator, const char *file_str, const char name_str[DAT__SZNAM], int *status);

/*=================================*/
/* hdsErase - Erase container file */
/*=================================*/

int
hdsErase_v5(HDSLoc **locator, int *status);

/*===============================================================*/
/* hdsEwild - End a wild card search for HDS container files     */
/*===============================================================*/

int
hdsEwild_v5(HDSWild *iwld, int *status);

/*================================*/
/* hdsFlush - Flush locator group */
/*=================================*/

int
hdsFlush_v5(const char *group_str, int *status);

/*===============================*/
/* hdsFree - Free container file */
/*===============================*/

int
hdsFree_v5(const HDSLoc *locator, int *status);

/*==================================*/
/* hdsGroup - Enquire locator group */
/*==================================*/

int
hdsGroup_v5(const HDSLoc *locator, char group_str[DAT__SZGRP+1], int *status);

/*=========================================*/
/* hdsGtune - Get HDS tuning parameter     */
/*=========================================*/

int
hdsGtune_v5(const char *param_str, int *value, int *status);

/*=========================================*/
/* hdsGtune - Get HDS status integers      */
/*=========================================*/

int
hdsInfoI_v5(const HDSLoc* locator, const char *topic_str, const char *extra, int *result, int *status);

/*=================================*/
/* hdsLink - Link locator to group */
/*=================================*/

int
hdsLink_v5(HDSLoc *locator, const char *group_str, int *status);

/*================================*/
/* hdsLock - Lock container file */
/*================================*/

int
hdsLock_v5(const HDSLoc *locator, int *status);

/*====================================*/
/* hdsNew - Create new container file */
/*====================================*/

int
hdsNew_v5(const char *file_str, const char *name_str, const char *type_str, int ndim, const hdsdim dims[], HDSLoc **locator, int *status);

/*========================================*/
/* hdsOpen - Open existing container file */
/*========================================*/

int
hdsOpen_v5(const char *file_str, const char *mode_str, HDSLoc **locator, int *status);

/*===============================*/
/* hdsShow - Show HDS statistics */
/*===============================*/

int
hdsShow_v5(const char *topic_str, int *status);

/*===============================================*/
/* hdsState - Enquire the current state of HDS   */
/*===============================================*/

int
hdsState_v5(hdsbool_t *state, int *status);

/*============================*/
/* hdsStop - Close down HDS   */
/*============================*/

int
hdsStop_v5(int *status);

/*==============================*/
/* hdsTrace - Trace object path */
/*==============================*/

int
hdsTrace_v5(const HDSLoc *locator, int *nlev, char *path_str, char *file_str, int *status, size_t path_length, size_t file_length);

/*========================================*/
/* hdsTune - Set HDS tuning parameter     */
/*========================================*/

int
hdsTune_v5(const char *param_str, int value, int *status);

/*=================================================================*/
/* hdsWild - Perform a wild-card search for HDS container files   */
/*=================================================================*/

int
hdsWild_v5(const char *fspec, const char *mode, HDSWild **iwld, HDSLoc **locator, int *status);

/*=================================================================*/
/*  Deprecated routines!                                           */
/*=================================================================*/

/*========================================*/
/* datConv - Enquire conversion possible? */
/*========================================*/

int
datConv_v5(const HDSLoc *locator, const char *type_str, hdsbool_t *conv, int *status);

/*=====================================================*/
/* hdsClose - Close container file (Obsolete routine!) */
/*=====================================================*/

int
hdsClose_v5(HDSLoc **locator, int *status);


/*===================================================================*/
/* hdsFind - Find an object (Fortran routine, requires hdsf library) */
/*===================================================================*/

int
hdsFind_v5(const HDSLoc *locator1, const char *name, const char *mode, HDSLoc **locator2, int *status);


/* STAR_HDS_V5_H_INCLUDED */
#endif
