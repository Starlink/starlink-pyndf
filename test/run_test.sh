#!/bin/bash
set -e


# Find python path
pybinary=`which python`
pydir=`dirname $pybinary`
ls -lhtr $pydir/../lib/python2.7/site-packages/starlink/
ldd $pydir/../lib/python2.7/site-packages/starlink/hds.so
ldd $pydir/../lib/python2.7/site-packages/starlink/ndf.so



# Relative path to this script.
SCRIPTPATH=`dirname $0`
echo $SCRIPTPATH

export HDS_VERSION=5
pytest -v $SCRIPTPATH

export HDS_VERSION=4
pytest -v $SCRIPTPATH
unset HDS_VERSION
