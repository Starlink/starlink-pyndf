#!/bin/bash
set -e

# Relative path to this script.
SCRIPTPATH=`dirname $0`

export HDS_VERSION=5
pytest -v $SCRIPTPATH

export HDS_VERSION=4
pytest -v $SCRIPTPATH
unset HDS_VERSION
