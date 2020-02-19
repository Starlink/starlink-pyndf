/*
*  Name:
*    vecUBtox.c

*  Purpose:
*    This file expands the generic C code held in vecUBtox.cgen to provide
*    the required type-specific implementations which can be called by
*    other functions.

*  Notes:
*    - This file is generated automatically at build time (see
*    Makefile.am)
*/

#include "prm_par.h"
#include "cgeneric.h"

#define CGEN_CODE_TYPE CGEN_DOUBLE_TYPE
#include "cgeneric_defs.h"
#include "vecUBtox.cgen"
#undef CGEN_CODE_TYPE

#define CGEN_CODE_TYPE CGEN_FLOAT_TYPE
#include "cgeneric_defs.h"
#include "vecUBtox.cgen"
#undef CGEN_CODE_TYPE

#define CGEN_CODE_TYPE CGEN_INT_TYPE
#include "cgeneric_defs.h"
#include "vecUBtox.cgen"
#undef CGEN_CODE_TYPE

#define CGEN_CODE_TYPE CGEN_WORD_TYPE
#include "cgeneric_defs.h"
#include "vecUBtox.cgen"
#undef CGEN_CODE_TYPE

#define CGEN_CODE_TYPE CGEN_UWORD_TYPE
#include "cgeneric_defs.h"
#include "vecUBtox.cgen"
#undef CGEN_CODE_TYPE

#define CGEN_CODE_TYPE CGEN_BYTE_TYPE
#include "cgeneric_defs.h"
#include "vecUBtox.cgen"
#undef CGEN_CODE_TYPE

#define CGEN_CODE_TYPE CGEN_UBYTE_TYPE
#include "cgeneric_defs.h"
#include "vecUBtox.cgen"
#undef CGEN_CODE_TYPE

#define CGEN_CODE_TYPE CGEN_INT64_TYPE
#include "cgeneric_defs.h"
#include "vecUBtox.cgen"
#undef CGEN_CODE_TYPE

