from libc.stdint cimport uint32_t, int64_t

cdef extern from "hds_types.h":
    ctypedef struct HDSLoc:
        pass
    ctypedef int64_t hdsdim;
    ctypedef int hdsbool_t;


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




# Use anonymous enums to define constants -- from cython readthedocs language basics.
cdef extern from "prm_par.h":
     cdef enum:
         VAL__BADUB
         VAL__BADB
         VAL__BADUW
         VAL__BADW
         VAL__BADI
         VAL__BADK
         VAL__BADR
         VAL__BADD


cdef extern from "dat_par.h":
    cdef enum:
        DAT__SZTYP
        DAT__SZNAM
        DAT__MXDIM

cdef extern from "err_par.h":
    cdef enum:
        ERR__SZPAR
        ERR__SZMSG

cdef extern from "sae_par.h":
    cdef enum:
        SAI__OK

cdef extern from "hds.h":
    int datAnnul(HDSLoc **locator, int *status);
    int datCell(const HDSLoc *locator1, int ndim, const hdsdim subs[], HDSLoc **locator2, int *status);
    int datClen(const HDSLoc *locator, size_t *clen, int *status);
    int datFind(const HDSLoc *locator1, const char *name_str, HDSLoc **locator2, int *status);
    int datGet(const HDSLoc *locator, const char *type_str, int ndim, const hdsdim dims[], void *values,
               int *status);
    int datIndex(const HDSLoc *locator1, int index, HDSLoc **locator2, int *status);
    int datLen(const HDSLoc *locator, size_t *len, int *status);
    int datName(const HDSLoc *locator, char name_str[], int *status);
    int datNcomp(const HDSLoc *locator, int *ncomp, int *status);
    int datNew(const HDSLoc *locator, const char *name_str, const char *type_str, int ndim,
               const hdsdim dims[], int *status);
    int datPut(const HDSLoc *locator, const char *type_str, int ndim, const hdsdim dims[],
               const void *values, int *status);
    int datPutC(const HDSLoc *locator, int ndim, const hdsdim dims[], const char string[],
                size_t string_length, int *status);
    int datShape(const HDSLoc *locator, int maxdim, hdsdim dims[], int *actdim, int *status);
    int datState(const HDSLoc *locator, hdsbool_t *state, int *status);
    int datStruc(const HDSLoc *locator, hdsbool_t *struc, int *status);
    int datType(const HDSLoc *locator, char type_str[], int *status);
    int datValid(const HDSLoc *locator, hdsbool_t *valid, int *status);

    int hdsNew(const char *file_str, const char *name_str, const char *type_str, int ndim,
                const hdsdim dims[DAT__MXDIM], HDSLoc **locator, int *status);
    int hdsOpen(const char *file_str, const char *mode_str, HDSLoc **locator, int *status);
    int hdsTrace(const HDSLoc *locator, int *nlev, char *path_str, char *file_str, int *status,
                 size_t path_length, size_t file_length);





