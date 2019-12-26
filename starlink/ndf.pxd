from libc.stdint cimport uint32_t, int64_t
#from cpython.ref cimport PyObject

cdef extern from "hds_types.h":
    ctypedef struct HDSLoc:
        pass
    ctypedef int64_t hdsdim;
    ctypedef int hdsbool_t;


cdef extern from "ndf_types.h":
    cdef enum:
        NDF__SZTYP
        NDF__SZMMD
        NDF__NOID
        NDF__NOPL
        NDF__MXDIM

cdef extern from "sae_par.h":
    cdef enum:
        SAI__OK

cdef extern from "sai_err.h":
    cdef enum:
        SAI__ERROR

cdef extern from "err_par.h":
    cdef enum:
        ERR__SZPAR
        ERR__SZMSG

cdef extern from "dat_par.h":
    cdef enum:
        DAT__FILNF
        DAT__SZTYP

cdef extern from "merswrap.h":
    void errBegin(int *status);
    void errAnnul(int *status);
    void errEnd(int *status);
    void errLoad(char *param,
              int param_length,
              int *parlen,
              char *opstr,
              int opstr_length,
              int *oplen,
              int *status );

cdef extern from "ast.h":
    ctypedef struct AstFrameSet:
        pass
    ctypedef struct AstObject:
        pass
    cdef:
       void astBegin()
       void astEnd()
       void astAnnul( AstObject *)
       void* astFree(void *)
       char *astToString( AstObject *)
       #PyObject* astFromString(const char * string)

#cdef extern from "star/pyast.h":
#    PyObject* PyAst_FromString(const char * string)

cdef extern from "ndf.h":
    void ndfAnnul( int *indf, int *status );

    void ndfBegin( );

    void ndfBound( int indf,
                   int ndimx,
                   int64_t  lbnd[NDF__MXDIM],
                   int64_t * ubnd,
                   int *ndim,
                   int *status );

    void ndfCget( int indf,
                  const char *comp,
                  char *value,
                  size_t value_length,
                  int *status );

    void ndfClen( int indf,
                  const char *comp,
                  size_t *length,
                   int *status );

    void ndfCput( const char *value,
                  int indf,
                  const char *comp,
                  int *status );

    void ndfDim( int indf,
                 int ndimx,
                 const hdsdim dim[],
                 int *ndim,
                 int *status );

    void ndfEnd( int *status );

    void ndfGtwcs( int indf,
                   AstFrameSet **iwcs,
                   int *status );


    void ndfOpen( const HDSLoc * loc,
                   const char *name,
                   const char *mode,
                   const char *stat,
                   int *indf,
                   int *place,
                   int *status );

    void ndfMap( int indf,
                 const char *comp,
                 const char *type,
                 const char *mmod,
                 void *pntr[],
                 size_t *el,
                 int *status );

    void ndfNew( const char *ftype,
                 int ndim,
                 const hdsdim lbnd[],
                 const hdsdim ubnd[],
                 int *place,
                 int *indf,
                 int *status );

    void ndfReset( int indf,
                   const char *comp,
                   int *status );

    void ndfSize( int indf,
                  size_t *npix,
                  int *status );

    void ndfState( int indf,
                   const char *comp,
                   int *state,
                   int *status );


    void ndfType( int indf,
                  const char *comp,
                  char *type,
                  size_t type_length,
                  int *status );


    void ndfUnmap( int indf,
                   const char *comp,
                   int *status );

    void ndfXloc( int indf,
                   const char *xname,
                   const char *mode,
                   HDSLoc ** loc,
                   int *status );

    void ndfXname( int indf,
                   int n,
                   char *xname,
                   size_t xname_length,
                   int *status );

    void ndfXnew( int indf,
                  const char *xname,
                  const char *type,
                  int ndim,
                  hdsdim dim[],
                  HDSLoc **loc,
                  int *status );

    void ndfXnumb( int indf,
                   int *nextn,
                   int *status );

    void ndfXstat( int indf,
                   const char *xname,
                   int *there,
                   int *status );


