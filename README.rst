starlink.hds
============

starlink.hds is a Python extension of the C-based Starlink HDS
library, allowing reading and writing of Starlink HDS format files
inside Python. It does not require the Starlink software suite to be
installed on your system. It also uses numpy, and it requires a c
compiler to built.

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
