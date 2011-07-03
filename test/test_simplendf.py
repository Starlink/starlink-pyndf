import unittest
import starlink.ndf as ndf
import starlink.hds as hds
import numpy
import os.path
import os

class TestSimpleNDF(unittest.TestCase):

    def setUp(self):
        ndf.ndf_begin()
        self.testndf = 'testme.sdf'

    def tearDown(self):
        ndf.ndf_end()
        os.remove( self.testndf )


    def test_simplenew(self):
        # okay we have all the data, time to open us up an ndf
        indf,place = ndf.ndf_open(self.testndf,'WRITE','NEW')
        newindf = ndf.ndf_new(indf,place,'_REAL',2,
                              numpy.array([0,0]),numpy.array([4,4]))

        # map primary data to make sure NDF does not complain
        ptr,el = ndf.ndf_map(newindf,'DATA','_REAL','WRITE')

        # make sure we got a file
        self.assertTrue( os.path.exists( self.testndf ), "Test existence of NDF file" )

    def test_newwithwrite(self):
        # okay we have all the data, time to open us up an ndf
        indf,place = ndf.ndf_open(self.testndf,'WRITE','NEW')
        newindf = ndf.ndf_new(indf,place,'_REAL',2,
                              numpy.array([0,0]),numpy.array([4,4]))

        # create PAMELA extension
        loc = ndf.ndf_xnew(newindf,'PAMELA','STRUCT')

        name = hds.dat_name(loc)
        self.assertEqual( name, "PAMELA" )

        ccd = numpy.zeros([5,5])

        # map primary data
        ptr,el = ndf.ndf_map(newindf,'DATA','_REAL','WRITE')
        ndf.ndf_numpytoptr(ccd,ptr,el,'_REAL')

        # shut down ndf system
        ndf.ndf_annul(newindf)

        # make sure we got a file
        self.assertTrue( os.path.exists( self.testndf ), "Test existence of NDF file" )

if __name__ == "__main__":
    unittest.main()

"""
License
=======

Copyright 2011 Tim Jenness
All Rights Reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
