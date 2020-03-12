starlink-pyndf documentation
============================

This package installs a standalone python interface to the Starlink
NDF and HDS libraries, allowing users to read and write NDF and HDS
files, without having the Starlink software suite installed. It
supports both HDS version 4 and version 5 (HDF5 based) For more
information on the HDS library and data format, please see
http://www.starlink.ac.uk/docs/sun92.htx/sun92.html , for the NDF
library and data format see
http://www.starlink.ac.uk/docs/sun33.htx/sun33.html and for
information on Starlink please see http://starlink.eao.hawaii.edu .

It also includes a higher level interface to NDF objects in the
`starlink.ndfpack` module. This interface predates the modern astropy
conventions, and is not currently under active development.

You will need to have the `numpy` package and the python interface to the
Starlink Ast World Coordinate System  installed (`starlink.pyast`: see
http://starlink.github.io/starlink-pyast/pyast.html).

You can install this package with

`pip install starlink-pyndf`

Or view the source at:

http://github.com/Starlink/starlink-pyndf


Using the software
******************


`starlink.ndf`
--------------


Import the module as:

>>> from starlink import ndf

In order to ensure accurate error handling and annuling of NDF objects
when you are finished, it is recommended that you call:

>>> ndf.begin()

before you start using the ndf module, and

>>> ndf.end()

afterwards.

You can open an existing NDF file and return an NDF object as:

>>> myndf = ndf.open('~/mask_trim.sdf', 'UPDATE')

The mode can be 'UPDATE', 'READ', 'WRITE' or 'NEW'.

You can get the NDF Pixel bounds in (z,y,x) format with:

>>> bounds = myndf.bound()


The text attributes of the NDF are viewable as:
>>> print(myndf.title)
>>> print(myndf.label)
>>> print(myndf.units)

You can get the WCS information as an AST frameset with:
>>> wcs = myndf.gtwcs()

You can read the values from an array  (e.g. DATA or VARIANCE) with:
>>> data = myndf.read('DATA')

which will return a numpy array with the appropriate type.

You can map an array (e.g. DATA or VARIANCE) with

>>> mapped = myndf.map('DATA', '_DOUBLE', 'WRITE')

And then update the values from a new numpy array with

>>> mapped.numpytondf(newdata)

You can then unmap the access with:

>>> mapped.unmap()

When you are finished with an NDF object, you can close it with:

>>> myndf.annul()

You can also access NDF extensions through the `starlink.ndf.xname`,
`starlink.ndf.xnumb`, `starlink.ndf.xstat` and `starlink.ndf.xloc`
methods.

`starlink.hds`
--------------

Import the module as:

>>> from starlink import hds

You can open an existing HDS file and return an HDS locator object as:

>>> hdslocator = hds.open('~/mask_trim.sdf', 'READ')
>>> print(hdslocator)
</home/sgraves/a20160513_00039_01_0001.sdf.A20160513_00039.DATA_ARRAY>

The hdslocator object can tell you what type of object it is:

>>> print(hdslocator.type)
NDF

Its name is:

>>> print(hdslocator.name)
'A20160513_00039'

You can see if it is a structure:

>>> print(hdslocator.struc)
True

If it is a structure, it will have subcomponents. You can see how many
of them there are with the 'ncomp' attribute, and index them with the
index method:

>>> print(hdslocator.ncomp)
6
>>> for i in range(hdslocator.ncomp):
        print(hdslocator.index(i))
</home/sgraves/a20160513_00039_01_0001.sdf.A20160513_00039.DATA_ARRAY>
</home/sgraves/a20160513_00039_01_0001.sdf.A20160513_00039.HISTORY>
</home/sgraves/a20160513_00039_01_0001.sdf.A20160513_00039.MORE>
</home/sgraves/a20160513_00039_01_0001.sdf.A20160513_00039.UNITS>
</home/sgraves/a20160513_00039_01_0001.sdf.A20160513_00039.LABEL>
</home/sgraves/a20160513_00039_01_0001.sdf.A20160513_00039.WCS>

You can also get the locator for a named subcomponent using the 'find'
command

>>> labelloc = hdslocator.find('LABEL')
>>> print(label)
</home/sgraves/a20160513_00039_01_0001.sdf.A20160513_00039.LABEL>

The label has struc=False, and its type is:

>>>print(labelloc.type)
_CHAR*57

Indicating that it is a string. The labelloc.shape attribute is None,
indicating that is not an array of strings but just one string.


You can get the value of component with locator.get:

>>> label = labelloc.get()
>>> print(label)
b'T%s60+%v30+A%^50+%<20+*%+   corrected antenna temperature'

Note that on Python 3 this will return a bytestring if the component
is a character string, which will need to be decoded, normally into
ascii characters.

(This label looks odd as it contains Starlink formatting characters to
print the T*A label correctly on a Starlink plot).


Some data files contain arrays rather than single values. The `shape`
attribute of the locator will tell you this. E.g. looking at the WCS
DATA (which in this case is an array of strings representing the AST
representation of the NDF files WCS)

>>> wcsstring = hdslocator.find('WCS').find('DATA').get() >>>
print(wcsstring.shape) (224, )

You can still get the value with the normal `.get()` function:

>>> wcs = wcsstring.get()


You can also get numeric types from HDS files: loking at the data
array of an NDF file for example:

>>> # Find the component named 'DATA_ARRAY' at the top level of an NDF
>>> datalocator = hdslocator.find('DATA_ARRAY')
>>> print(datalocator)
</home/sgraves/a20160513_00039_01_0001.sdf.A20160513_00039.DATA_ARRAY>

The data array consists of 2 components:

>>> print(datalocator.ncomp)
2

These are the data array itself, (named 'DATA') and the data
origin. To get the data_array as a numpy array, do:

>>> arrayloc = datalocator.find('DATA')
>>> data_array = arrayloc.get()


Some data arrays will have bad pixels. These are indicated by the
presence of a pixel value corresponding to the HDS 'BAD' value for
that data type. You can create a mask for a data file indicating where
pixels are bad with the following code:

First find out the HDS type of the data array ('_REAL', '_DOUBLE', '_INT' etc.)

>>> type_ = arrayloc.type

Now find out what the bad value is for that specific type:

>>> badvalue = hds.getbadvalue(type_)

Now use normal numpy features to either a) create a masked array (recommended)

>>> mask = data_array == badvalue
>>> masked_data = np.ma.MaskedArray(data_array, mask=mask)

or b) set the BAD values to be NaN (this is not possible if the data
are integers):

>>> import numpy as np
>>> data_array[data_array == badvalue] = np.nan

Some HDS components are more complicated. The HISTORY.RECORDS
component of an NDF file (if it exists), consists of arrays of
records.

These types of arrays of objects are more complicated than simple
arrays of the basic numeric or string data types: instead each element
of the array consists of complex HDS structures with multiple
components themselves

>>> records = hdslocator.find('HISTORY').find('RECORDS')

They have a shape:

>>> print(records.shape)
array([10], dtype=int32)

But they are also indicated as a structure:

>>> print(records.struc)
True

However, if you try and look at their 'ncomp' they will inidicate an error:

>>> print(records.ncomp)
error: DAT_NCOMP: Error enquiring the number of components in an HDS structure.

Instead, you must use the 'locator.cell' function to access them.
In this case, the first record is at:

>>> print(records.cell([0])
</home/sgraves/a20160513_00039_01_0001.sdf.A20160513_00039.HISTORY.RECORDS(1)>


If the shape of the array indicated multiple dimensions, you would
need to give the index seperately for each index. E.g. to index the
first element of a 3-dimensional structure you would use:

>>> structure_array.cell([0,0,0])

Note that the objects returned by the `cell` method are locators, and
in general the contents are normally a strucutred set of HDS
components not scalars. You would use the index,find,ncomp,get,cell
etc methods/attributes to get specific values from them, just as
described above.


Contents:

.. toctree::
   :maxdepth: 4

   starlink.ndf
   starlink.hds
   starlink.ndfpack


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

