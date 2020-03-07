starlink-pyndf
==============

Starlink-pyndf contains starlink.hds, starlink.ndf and starlink.ndfpack modules.

starlink.hds and starlink.ndf are Python extension of the C-based Starlink HDS and NDF
libraries, allowing reading and writing of Starlink HDS  and NDF format files
inside Python. It can read and write version 4 and version 5 (HDF5
based) HDS files, and supports 64 bit indexing in HDS/NDF files. It does not require the Starlink software suite to be
installed on your system. It will build the HDF5 library itself and
does not require a separate installation of that. It currently supports Python 2.7+.

starlink.ndfpack provides high level access to an NDF object.

For more information on HDS see:
http://www.starlink.ac.uk/docs/sun92.htx/sun92.html

For information on NDF see:
http://starlink.eao.hawaii.edu/docs/sun33.htx/sun33.html

or for information on Starlink itself see
http://starlink.eao.hawaii.edu/starlink

Installation
************
It can be installed with pip, or directly build from source with

python setup.py install

or

python setup.py install --prefix=<your specific installation directory>


Updating library versions (for maintainers)
*******************************************

If it necessary to update the libraries, you will need to ensure that any required built files have also been updated and included.
