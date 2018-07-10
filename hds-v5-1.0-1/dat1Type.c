/*
*+
*  Name:
*     dat1Type

*  Purpose:
*     Enquire the type of an object (enum version)

*  Language:
*     Starlink ANSI C

*  Type of Module:
*     Library routine

*  Invocation:
*     hdstype_t dat1Type( const HDSLoc *locator, int * status );

*  Arguments:
*     locator = const HDSLoc * (Given)
*        Object locator
*     status = int* (Given and Returned)
*        Pointer to global status.

*  Returned Value:
*     type = hdstype_t
*        Enum indicating the HDS type associated with this
*        locator. Returns HDSTYPE_NONE on error.

*  Description:
*     Enquire the type of an object and returns that type as an enum.
*     use datType() to return a string form that will also return types
*     of structures and details of string length.

*  Authors:
*     TIMJ: Tim Jenness (Cornell)
*     {enter_new_authors_here}

*  History:
*     2014-08-27 (TIMJ):
*        Initial version
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

#include "hdf5.h"

#include "star/one.h"
#include "ems.h"
#include "sae_par.h"

#include "hds1.h"
#include "dat1.h"
#include "hds.h"
#include "dat_err.h"

hdstype_t
dat1Type( const HDSLoc *locator, int * status ) {

  hid_t h5type = 0;
  hdstype_t thetype = HDSTYPE_NONE;

  if (*status != SAI__OK) return thetype;

  /* if this is a group locator we can return straightaway */
  if (dat1IsStructure(locator, status)) return HDSTYPE_STRUCTURE;

  CALLHDF( h5type,
           H5Dget_type( locator->dataset_id ),
           DAT__HDF5E,
           emsRep("dat1Type_1", "datType: Error obtaining data type of dataset", status)
           );

  thetype = dau1HdsType( h5type, status );

 CLEANUP:
  return thetype;
}

