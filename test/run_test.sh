#!/bin/bash
set -e

# Absolute path to this script. /home/user/bin/foo.sh
SCRIPT=$(readlink -f $0)
# Absolute path this script is in. /home/user/bin
SCRIPTPATH=`dirname $SCRIPT`

export HDS_VERSION=5
pytest -v $SCRIPTPATH
export HDS_VERSION=4
pytest -v $SCRIPTPATH
unset HDS_VERSION
