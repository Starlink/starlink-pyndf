#!/bin/bash
set -e
export HDS_VERSION=5
pytest -v
export HDS_VERSION=4
pytest -v
unset HDS_VERSION
