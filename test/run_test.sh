#!/bin/bash
set -e
export HDS_VERSION=5
nosetests -v
export HDS_VERSION=4
nosetests -v
unset HDS_VERSION
