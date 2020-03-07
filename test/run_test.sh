#!/bin/bash
set -e


# Find python path
pybinary=`which python`
pydir=`dirname $pybinary`

# Relative path to this script.
SCRIPTPATH=`dirname $0`
echo $SCRIPTPATH
cd $SCRIPTPATH

export HDS_VERSION=5
pytest -v $SCRIPTPATH

export HDS_VERSION=4
pytest -v $SCRIPTPATH
unset HDS_VERSION
