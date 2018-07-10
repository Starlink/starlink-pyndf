
#if !defined( HDS1_INCLUDED )    /* hds1.h already included?                */
#define HDS1_INCLUDED 1

#define HDS_INTERNAL_INCLUDES 1

/* Memory Allocation Routines */
/* ===========================*/
/* Define macros here for to allow us to easily fallback to native free/malloc */
#include "star/mem.h"
#define MEM_MALLOC  starMalloc
#define MEM_FREE    starFree
#define MEM_REALLOC starRealloc
#define MEM_CALLOC  starCalloc

/* EMS wrapper routines:*/
/* =====================*/
#include "hds_types.h"
void dat1emsSetHdsdim( const char * token, hdsdim value );




#endif
