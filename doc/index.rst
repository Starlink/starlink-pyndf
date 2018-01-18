starlink-pyhds documentation
============================

This package installs a standalone python interface to the Starlink
HDS library, allowing users to read and write Starlink HDS-V4 files
without having the Starlink software suite installed. For more
information on the HDS library and data format, please see
http://www.starlink.ac.uk/docs/sun92.htx/sun92.html , and for information on Starlink please see http://starlink.eao.hawaii.edu .

This packagedoes not include the NDF library, although it can open and
read NDF format files (extension .sdf), and could be used to write
them if the user chooses to take care of following the data from the
file. If you require a python interface to the NDF library, please see
the starlink-pyndf package (at
https://github.com/timj/starlink-pyndf ), although note that
requires an existing Starlink software installation on your computer.



Using the software
******************
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

   starlink.hds


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

