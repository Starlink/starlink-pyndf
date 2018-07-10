/*
*+
*  Name:
*     hdsShow

*  Purpose:
*     Show HDS statistics

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hdsShow(const char *topic_str, int  *status);

*  Arguments:
*     topic_str = const char * (Given)
*        Topic name.
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Description:
*     Display statistics about the specified topic on the standard output.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  Notes:
*     The following topics are supported:
*     - FILES: List all open file objects.
*     - LOCATORS: List all open primitives and structure locators.

*  History:
*     2014-10-17 (TIMJ):
*        Initial version
*     2014-11-07 (TIMJ):
*        Add preliminary support for FILES and LOCATORS.
*     {enter_further_changes_here}

*  Copyright:
*     Copyright (C) 2014 Cornell University
*     All Rights Reserved.

*  Licence:
*     Redistribution and use in source and binary forms, with or
*     without modification, are permitted provided that the following
*     conditions are met:
*
*     - Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*
*     - Redistributions in binary form must reproduce the above
*       copyright notice, this list of conditions and the following
*       disclaimer in the documentation and/or other materials
*       provided with the distribution.
*
*     - Neither the name of the {organization} nor the names of its
*       contributors may be used to endorse or promote products
*       derived from this software without specific prior written
*       permission.
*
*     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
*     CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
*     INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
*     MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
*     CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
*     SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
*     LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
*     USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
*     AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
*     IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
*     THE POSSIBILITY OF SUCH DAMAGE.

*  Bugs:
*     {note_any_bugs_here}
*-
*/

#include <string.h>

#include "hdf5.h"

#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"

#include "dat_err.h"

int
hdsShow(const char *topic_str, int  *status) {

  hid_t *obj_id_list = NULL;

  if (*status != SAI__OK) return *status;

  if (strncasecmp( topic_str, "DAT", 3 ) == 0) {
    *status = DAT__FATAL;
    emsRep("hdsShow", "hdsShow: DATA reporting not yet implemented for HDF5",
           status);
  } else if (strncasecmp( topic_str, "FIL", 3 ) == 0 ||
             strncasecmp( topic_str, "LOC", 3 ) == 0 ) {
    hid_t *obj_id_list = NULL;
    ssize_t nobj = 0;
    ssize_t nret = 0;
    unsigned int types = 0;
    ssize_t i;
    hdsbool_t listfiles = 0;
    hdsbool_t listlocs = 0;

    /* Decide what we are interested in */
    switch (topic_str[0]) {
    case 'F':
    case 'f':
      types = H5F_OBJ_FILE;
      listfiles = 1;
      break;
    case 'L':
    case 'l':
      types = H5F_OBJ_DATASET | H5F_OBJ_GROUP;
      listlocs = 1;
      break;
    default:
      *status = DAT__WEIRD;
      emsRep("hdsShow_2", "hdsShow: Possible programming error in topic handling",
              status);
      goto CLEANUP;
    }

    /* First count how many objects we need to allocate */
    CALLHDFE( ssize_t,
              nobj,
              H5Fget_obj_count( H5F_OBJ_ALL, types ),
              DAT__HDF5E,
              emsRep("hdsShow_3", "Unable to query number of active HDF5 objects",
                     status )
              );

    if (nobj > 0) {

      /* Allocate memory */
      obj_id_list = MEM_CALLOC( nobj, sizeof(*obj_id_list) );

      /* Then actually get them */
      CALLHDFE( ssize_t,
                nret,
                H5Fget_obj_ids( H5F_OBJ_ALL, types, nobj, obj_id_list ),
                DAT__HDF5E,
                emsRep("hdsShow_4", "Error retrieving active HDF5 objects",
                       status )
                );

      if (nobj != nret) {
        /* Internal weirdness */
        *status = DAT__WEIRD;
        emsRep("hdsShow_5", "hdsShow: Number of objects expected != number returned",
               status );
        goto CLEANUP;
      }
      printf("hdsShow: Obtained %d relevant HDF5 object%s:\n", (int)nobj,
             (nobj == 1 ? "" : "s"));
      /* Naively report on each object */
      for (i=0; i<nobj; i++) {
        H5I_type_t objtype;
        hid_t obj_id;
        obj_id = obj_id_list[i];
        objtype = H5Iget_type( obj_id );
        if ( objtype == H5I_FILE ) {
          unsigned intent = 0;
          char * name_str = NULL;
          const char * intent_str = NULL;

          H5Fget_intent( obj_id, &intent );
          if (intent == H5F_ACC_RDONLY) {
            intent_str = "R";
          } else if (intent == H5F_ACC_RDWR) {
            intent_str = "U";
          } else {
            intent_str = "Err";
          }
          name_str = dat1GetFullName( obj_id, 1, NULL, status );
          printf("File: %s [%s] (%d)\n", name_str, intent_str, obj_id );
          if (name_str) MEM_FREE(name_str);
        } else if ( objtype == H5I_GROUP || objtype == H5I_DATASET ) {
          char * name_str = NULL;
          name_str = dat1GetFullName( obj_id, 0, NULL, status );
          printf("%s: %s\n", (objtype == H5I_GROUP ? "Group" : "Dataset"), name_str );
          if (name_str) MEM_FREE(name_str);
        } else if ( objtype == H5I_ATTR ) {
          printf("Unexpectedly got an open attribute\n");
        } else if ( objtype == H5I_DATATYPE ) {
          printf("Unexpectedly got an open datatype\n");
        } else if ( objtype == H5I_DATASPACE ) {
          printf("Unexpectedly got an open dataspace\n");
        } else {
          /* Unexpected */
          printf("Unexpectedly got a bad data type\n");
        }
      }
    } else {
      printf("hdsShow: Obtained 0 relevant HDF5 objects\n");
    }

    /* And internal HDS report */
    if (listfiles || listlocs) hds1ShowFiles( listfiles, listlocs, status );
  } else {
    *status = DAT__FATAL;
    emsRepf("hdsShow", "hdsShow: Do not understand topic '%s'",
            status, topic_str);
  }

 CLEANUP:
  if (obj_id_list) MEM_FREE(obj_id_list);
  return *status;
}
