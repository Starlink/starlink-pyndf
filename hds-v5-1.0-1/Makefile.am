## Process this file with automake to produce Makefile.in

## bin_PROGRAMS = hdsdump
## dist_bin_SCRIPTS = hds_link hds_link_adam
noinst_PROGRAMS = make-hds-types

# libhdsf is for fortran interface
lib_LTLIBRARIES = libhds_v5.la

TESTS = hdsTest
check_PROGRAMS = hdsTest

libhds_v5_la_SOURCES = \
	$(PUBLIC_INCLUDES) \
	$(PUBLIC_CINCLUDES) \
	$(include_MESSAGES) \
	$(PRIVATE_INCLUDES) \
	$(C_ROUTINES)

libhds_v5_la_LIBADD = `starmem_link` `ems_link` `one_link` -lhdf5

# Make all library code position independent. This is handy for creating
# shareable libraries from the static ones (Java JNI libraries).
if !NOPIC
libhds_v5_la_CFLAGS = $(AM_CFLAGS) -prefer-pic
endif

make_hds_types_SOURCES = make-hds-types.c

hdsTest_SOURCES = hdsTest.c
hdsTest_LDADD = libhds_v5.la

## hds_test_prm_SOURCES = hds_test_prm.c
## hds_test_prm_LDADD = libhds.la `ems_link` `cnf_link` `cnf_link`

cincludedir = $(includedir)/star
cinclude_HEADERS = $(PUBLIC_CINCLUDES)
include_HEADERS = $(PUBLIC_INCLUDES)
noinst_MESSAGES = dat_err.h

# dat_par.h was previously released so must go in the root include dir
PUBLIC_CINCLUDES = hds_v5.h

PRIVATE_INCLUDES = \
	dat_par.h hds_types.h \
	hds1_types.h \
	hds_fortran.h \
	dat1.h \
	$(noinst_MESSAGES)

C_ROUTINES = \
	$(PUBLIC_C_ROUTINES) \
	$(PRIVATE_C_ROUTINES)

PUBLIC_C_ROUTINES = \
datAlter.c \
datAnnul.c \
datBasic.c \
datCcopy.c \
datCctyp.c \
datCell.c \
datChscn.c \
datClen.c \
datClone.c \
datCopy.c \
datCoerc.c \
datDrep.c \
datErase.c \
datErmsg.c \
datExportFloc.c \
datFind.c \
datGet.c \
datGet1C.c \
datGetVC.c \
datImportFloc.c \
datIndex.c \
datLen.c \
datMap.c \
datMapN.c \
datMould.c \
datMove.c \
datMsg.c \
datName.c \
datNcomp.c \
datNew.c \
datParen.c \
datPrec.c \
datPrim.c \
datPrmry.c \
datPut.c \
datPut1C.c \
datPutVC.c \
datRef.c \
datRefct.c \
datRenam.c \
datReset.c \
datRetyp.c \
datShape.c \
datSize.c \
datSlice.c \
datState.c \
datStruc.c \
datTemp.c \
datThere.c \
datType.c \
datUnmap.c \
datValid.c \
datVec.c \
datLock.c \
datLocked.c \
datNolock.c \
datUnlock.c \
hdsCopy.c \
hdsErase.c \
hdsEwild.c \
hdsDimC2F.c \
hdsDimF2C.c \
hdsFree.c \
hdsGroup.c \
hdsInfoI.c \
hdsLock.c \
hdsNew.c \
hdsOpen.c \
hdsShow.c \
hdsState.c \
hdsStop.c \
hdsTrace.c \
hdsWild.c \
datConv.c \
hdsClose.c \
datget0x.c \
datgetx.c \
datmapx.c \
datnew0.c \
datnew1.c \
datput0.c \
datput1.c \
hdstuning.c \
hdsgroups.c

PRIVATE_C_ROUTINES = \
dat1AllocLoc.c \
dat1Annul.c \
dat1Coords2CellName.c \
dat1CreateStructureCell.c \
dat1CvtChar.c \
dat1CvtLogical.c \
dat1DumpLoc.c \
dat1emsSetHdsdim.c \
dat1EncodeSubscript.c \
dat1EraseHandle.c \
dat1ExportDims.c \
dat1FixNameCell.c \
dat1FreeHandle.c \
dat1FreeLoc.c \
dat1GetAttr.c \
dat1GetAttrBool.c \
dat1GetAttrHdsdims.c \
dat1GetAttrInt.c \
dat1GetAttrString.c \
dat1GetBounds.c \
dat1GetDataDims.c \
dat1Getenv.c \
dat1GetFullName.c \
dat1GetParentID.c \
dat1GetStructureDims.c \
dat1Handle.c \
dat1HandleLock.c \
dat1HandleMsg.c \
dat1H5EtoEMS.c \
dat1ImportDims.c \
dat1ImportFloc.c \
dat1Index2Coords.c \
dat1InitHDF5.c \
dat1IsStructure.c \
dat1NeedsRootName.c \
dat1New.c \
dat1NewPrim.c \
dat1RetrieveContainer.c \
dat1RetrieveIdentifier.c \
dat1SetAttr.c \
dat1SetAttrBool.c \
dat1SetAttrHdsdims.c \
dat1SetAttrInt.c \
dat1SetAttrString.c \
dat1SetStructureDims.c \
dat1Type.c \
dat1TypeInfo.c \
dau1CheckFileName.c \
dau1CheckName.c \
dau1CheckType.c \
dau1HdsType.c \
dau1Native2MemType.c \
dat1ValidateLocator.c \
hdstrack.c

hds_types.h: make-hds-types$(EXEEXT)
	./make-hds-types

hds1_types.h: make-hds-types$(EXEEXT)
	./make-hds-types

# Have to make sure that hds_types appears before DAT_PAR so that
# the include file generation routines are built in the correct order
BUILT_SOURCES = hds_types.h hds1_types.h \
	$(include_MESSAGES) $(noinst_MESSAGES)

CLEANFILES = dat_err.h \
	hds_types.h hds1_types.h

## dist_starnews_DATA = hds.news

EXTRA_DIST = dat_err.msg

