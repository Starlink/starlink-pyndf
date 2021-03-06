      subroutine cmp_mapv(struct, comp, type, mode, ptr, actval,
     :  status)
*+
*  Name:
*     CMP_MAPV

*  Purpose:
*     Map component as if it were a vector.

*  Language:
*     VAX Fortran

*  Invocation:
*     CALL CMP_MAPV(LOC, NAME, TYPE, MODE, PNTR, ACTVAL, STATUS)

*  Description:
*     This routine maps a primitive component of a structure for
*     reading, writing or updating, as if it were vectorized.
*     The number of values mapped is returned in the variable,
*     ACTVAL.

*  Arguments:
*     LOC=CHARACTER*(DAT__SZLOC)
*        Variable containing a locator associated with a structured
*        data object.
*     NAME=CHARACTER*(*)
*        Expression specifying the component name of a primitive
*        object contained in the structure.
*     TYPE=CHARACTER*(*)
*        Expression specifying the data type of the mapped values.
*        If the actual type of the data object differs from this,
*        then conversion will be performed in 'READ' and 'UPDATE'
*        modes.
*     MODE=CHARACTER*(*)
*        Expression specifying the mode in which the data are to be
*        mapped.  (Either 'READ', 'WRITE' or 'UPDATE'.)
*     PNTR=INTEGER
*        Variable to receive the virtual memory pointer for the
*        mapped values.   This can be used in conjunction with the
*        VAX Fortran "%VAL" construct.
*     ACTVAL=INTEGER
*        Variable to receive the number of values mapped.
*     STATUS=INTEGER
*        Variable holding the status value. If this variable is not
*        SAI__OK on input, the routine will return without action.
*        If the routine fails to complete, this variable will be
*        set to an appropriate error number.

*  Algorithm:
*     Check that the component is not already mapped.   Find
*     a slot in the Component Table.  Map in data values.
*     Update Component Table based on outcome.

*  Authors:
*     Jack Giddings (UCL::JRG)
*     {enter_new_authors_here}

*  History:
*     3-JAN-1982:  Original.  (UCL::JRG)
*     15-APR-1987:  Improved prologue layout (RAL::AJC)
*     {enter_further_changes_here}

*  Bugs:
*     {note_any_bugs_here}

*-

*  Global Constants:
      INCLUDE 'SAE_PAR'
      INCLUDE 'DAT_PAR'
      INCLUDE 'CMP_CONST'
      INCLUDE 'CMP_ERR'

*  Arguments Given:
      character*(*) struct		! Structure Locator
      character*(*) comp		! Component Name
      character*(*) type		! Access type
      character*(*) mode		! Access mode

*  Arguments Returned:
      integer ptr			! Address pointer
      integer actval			! Actual number of values
*    Status return :
      integer status			! Status Return

*  External References:
      logical chr_simlr			! Caseless string equality

*  Global Variables:
      INCLUDE 'CMP_CCT'

*  Local Variables:
      integer index			! Table index

*.


*    Allowed to execute ?
      if (status .ne. SAI__OK) then
         return
      endif

*    Initialised ?
      if (Cmpslp) then
         call cmp_activ(status)
         if (status .ne. SAI__OK) then
            return
         endif
      endif

*    Check that object is not already in the Component Table
      index = 1
      dowhile (index .le. Cmpcnt)
         if (.not. Cmpfre(index)) then
            if (struct .eq. Cmpstr(index)) then
               if (chr_simlr(comp, Cmpnam(index))) then
                  status = CMP__ISMAP
                  call cmp_erdsn(struct, comp, status)
                  return
               endif
            endif
         endif
         index = index + 1
      enddo

*    Find a free slot
      index = 1
      dowhile (index .le. CMP__MXCMP)
         if (index .gt. Cmpcnt) then
            goto 1
         elseif (Cmpfre(index)) then
            goto 1
         endif
         index = index + 1
      enddo
 1    continue

*    Check against table overflow
      if (index .gt. CMP__MXCMP) then
         status = CMP__FATAL
         call cmp_erdsn(struct, comp, status)
         return
      endif

*    Get component locator and map values
      call dat_find(struct, comp, Cmploc(index), status)
      if (status .ne. SAI__OK) then
         call cmp_erdsn(struct, comp, status)
      else
         call dat_mapv(Cmploc(index), type, mode, ptr, actval,
     :     status)
         if (status .ne. SAI__OK) then
            call dat_annul(Cmploc(index), status)
         else
            Cmpstr(index) = struct
            Cmpnam(index) = comp
            Cmpfre(index) = .false.
            if (index .gt. Cmpcnt) then
               Cmpcnt = Cmpcnt + 1
            endif
         endif
      endif

      end


