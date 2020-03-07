void ndfGtszD_( int indf, const char *comp, double *scale, double *zero, int *status );
void ndfPtszD_( double scale, double zero, int indf, const char *comp, int *status );
void ndfXgt0D_( int indf, const char *xname, const char *cmpt, double *value, int *status );
void ndfXpt0D_( double value, int indf, const char *xname, const char *cmpt, int *status );
void ndfGtszF_( int indf, const char *comp, float *scale, float *zero, int *status );
void ndfPtszF_( float scale, float zero, int indf, const char *comp, int *status );
void ndfXgt0F_( int indf, const char *xname, const char *cmpt, float *value, int *status );
void ndfXpt0F_( float value, int indf, const char *xname, const char *cmpt, int *status );
void ndfGtszI_( int indf, const char *comp, int *scale, int *zero, int *status );
void ndfPtszI_( int scale, int zero, int indf, const char *comp, int *status );
void ndfXgt0I_( int indf, const char *xname, const char *cmpt, int *value, int *status );
void ndfXpt0I_( int value, int indf, const char *xname, const char *cmpt, int *status );
void ndfGtszW_( int indf, const char *comp, short int *scale, short int *zero, int *status );
void ndfPtszW_( short int scale, short int zero, int indf, const char *comp, int *status );
void ndfXgt0W_( int indf, const char *xname, const char *cmpt, short int *value, int *status );
void ndfXpt0W_( short int value, int indf, const char *xname, const char *cmpt, int *status );
void ndfGtszUW_( int indf, const char *comp, unsigned short int *scale, unsigned short int *zero, int *status );
void ndfPtszUW_( unsigned short int scale, unsigned short int zero, int indf, const char *comp, int *status );
void ndfXgt0UW_( int indf, const char *xname, const char *cmpt, unsigned short int *value, int *status );
void ndfXpt0UW_( unsigned short int value, int indf, const char *xname, const char *cmpt, int *status );
void ndfGtszB_( int indf, const char *comp, char *scale, char *zero, int *status );
void ndfPtszB_( char scale, char zero, int indf, const char *comp, int *status );
void ndfXgt0B_( int indf, const char *xname, const char *cmpt, char *value, int *status );
void ndfXpt0B_( char value, int indf, const char *xname, const char *cmpt, int *status );
void ndfGtszUB_( int indf, const char *comp, unsigned char *scale, unsigned char *zero, int *status );
void ndfPtszUB_( unsigned char scale, unsigned char zero, int indf, const char *comp, int *status );
void ndfXgt0UB_( int indf, const char *xname, const char *cmpt, unsigned char *value, int *status );
void ndfXpt0UB_( unsigned char value, int indf, const char *xname, const char *cmpt, int *status );
void ndfGtszK_( int indf, const char *comp, int64_t *scale, int64_t *zero, int *status );
void ndfPtszK_( int64_t scale, int64_t zero, int indf, const char *comp, int *status );
void ndfXgt0K_( int indf, const char *xname, const char *cmpt, int64_t *value, int *status );
void ndfXpt0K_( int64_t value, int indf, const char *xname, const char *cmpt, int *status );

/* Now define the macros used by application code to invoke the appropriate
   functions, depending on whether the old or new interface is required. 
   Currently, there is no diffference betwen the two interfaces for the 
   functions defined in this file but this may change in the future. */

#define ndfGtszd ndfGtszD_
#define ndfGtszr ndfGtszF_
#define ndfGtszi ndfGtszI_
#define ndfGtszw ndfGtszW_
#define ndfGtszuw ndfGtszUW_
#define ndfGtszb ndfGtszB_
#define ndfGtszub ndfGtszUB_
#define ndfGtszk ndfGtszK_
#define ndfPtszd ndfPtszD_
#define ndfPtszr ndfPtszF_
#define ndfPtszi ndfPtszI_
#define ndfPtszw ndfPtszW_
#define ndfPtszuw ndfPtszUW_
#define ndfPtszb ndfPtszB_
#define ndfPtszub ndfPtszUB_
#define ndfPtszk ndfPtszK_
#define ndfXgt0d ndfXgt0D_
#define ndfXgt0r ndfXgt0F_
#define ndfXgt0i ndfXgt0I_
#define ndfXgt0w ndfXgt0W_
#define ndfXgt0uw ndfXgt0UW_
#define ndfXgt0b ndfXgt0B_
#define ndfXgt0ub ndfXgt0UB_
#define ndfXgt0k ndfXgt0K_
#define ndfXpt0d ndfXpt0D_
#define ndfXpt0r ndfXpt0F_
#define ndfXpt0i ndfXpt0I_
#define ndfXpt0w ndfXpt0W_
#define ndfXpt0uw ndfXpt0UW_
#define ndfXpt0b ndfXpt0B_
#define ndfXpt0ub ndfXpt0UB_
#define ndfXpt0k ndfXpt0K_

