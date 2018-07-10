starlink.hds
============

starlink.hds is a Python extension of the C-based Starlink HDS
library, allowing reading and writing of Starlink HDS format files
inside Python. It can read and write version 4 and version 5 (HDF5
based) HDSfiles. It does not require the Starlink software suite to be
installed on your system. It also requires numpy, and it requires a c
compiler to built. It will build the HDF5 library itself and
does not require a separate installation of that.

As the build procedure uses starlink-generated configure scripts to
build the hds-v4 and hds-v5 libraries, it may unfortunately require
that gfortran or similar is present on the build machine. However, the fortran libraries are not used here.

This was adapted from the starlink.hds packaged shipped inside
starlink-pyndf: https://github.com/timj/starlink-pyndf The primary
difference is that this package does not require a separate Starlink
installation as it directly builds the C HDS library (and
dependencies), and it does not include any NDF specific access.

For more information on HDS see:
http://www.starlink.ac.uk/docs/sun92.htx/sun92.html
or for information on Starlink itself see
http://starlink.eao.hawaii.edu/starlink

Installation
************
It is installed via.

python setup.py install

or

python setup.py install --prefix=<your specific installation directory>


Updating library versions (for maintainers)
*******************************************

If it necessary to update the libraries, you should do a 'make dist'
in a configured Starlink install, copying the tarball into here, and
then un-tarring the resulting distribution into this repo. Add the
files to git, and check if there are any missing files that were not
included in the dist tar ball. The normal Starlink build procedure
does not use the 'make dist' step, so it is fairly common for new
header files and similar to be missing. Please update the upstream
starlink Makefile.am as appropriate to fix it.
