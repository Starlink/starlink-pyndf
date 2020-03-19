import unittest
import starlink.ndf as ndf
import starlink.hds as hds
import numpy as np
import os.path
import os
import pathlib
fulldir = pathlib.Path(__file__).parent.absolute().as_posix()


class TestSimpleNDF(unittest.TestCase):

    def setUp(self):
        ndf.begin()
        self.testndf = 'testme.sdf'

    def tearDown(self):
        ndf.end()
        if os.path.isfile(self.testndf):
            os.remove(self.testndf)

    def test_simpleread(self):
        indf = ndf.open(os.path.join(fulldir, 'data', 'ndf_test.sdf'))
        self.assertEqual(indf.label, 'Signal')
        self.assertEqual(indf.units, 'counts')
        self.assertEqual(indf.title, 'Test Data')
        self.assertEqual(indf.xnumb, 1)
        self.assertEqual(indf.dim, [5, 3])

        # Xtension sutf
        # xnumb (property), xname, xloc,  xstat, xdel,  xnew
        self.assertEqual(indf.xname(0), 'FITS')
        self.assertEqual(indf.xstat('FITS'), True)
        self.assertEqual(indf.xstat('NONEXISTENT'), False)
        self.assertIsInstance(indf.xloc('FITS', 'READ'),
                              hds.HDSWrapperClass)

        # AXSIS  stuff
        # aform, astat, acget, anorm, amap,  aread,
        self.assertEqual(indf.astat('LABEL', 0), True)
        self.assertEqual(indf.astat('UNIT', 0), True)
        self.assertEqual(indf.astat('WIDTH', 0), False)
        self.assertEqual(indf.astat('CENTRE', 0), True)
        self.assertEqual(indf.astat('VARIANCE', 0), False)

        self.assertEqual(indf.astat('LABEL', 1), True)
        self.assertEqual(indf.astat('UNIT', 1), True)
        self.assertEqual(indf.astat('WIDTH', 1), False)
        self.assertEqual(indf.astat('CENTRE', 1), True)
        self.assertEqual(indf.astat('VARIANCE', 1), False)

        self.assertEqual(indf.acget('LABEL', 1), 'Right ascension')
        self.assertEqual(indf.acget('LABEL', 0), 'Declination')
        self.assertEqual(indf.acget('UNIT', 1), 'deg')
        self.assertEqual(indf.acget('UNIT', 0), 'deg')

        # Main ndf stuff
        history = indf.history()
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0].application.split()[0], 'HISSET')
        self.assertEqual(indf.type('DATA'), '_REAL')
        self.assertSequenceEqual(indf.bound(), [[1, 1], [5, 3]])

        data = indf.read('DATA')
        data_shouldbe = (np.ones([5, 3])*-3.4028235e+38).astype(np.float32)
        self.assertSequenceEqual(data.tolist(), data_shouldbe.tolist())

        #  cget, , new, map (and using it to change comp),
        #  annul, , state
        from starlink import Ast
        self.assertIsInstance(indf.gtwcs(), Ast.FrameSet)

        self.assertEqual(indf.state('LABEL'), True)
        self.assertEqual(indf.state('QUALITY'), False)
        with self.assertRaises(hds.StarlinkError):
            indf.state('NONEXISTANT')

        indf.annul()

    def test_simplenew(self):
        # okay we have all the data, time to open us up an ndf
        indf = ndf.open(self.testndf, 'WRITE', 'NEW')
        indf.new('_REAL', 2,
                 np.array([0, 0]), np.array([4, 4]))

        # map primary data to make sure NDF does not complain
        ndfmap = indf.map('DATA', '_REAL', 'WRITE')
        self.assertEqual(ndfmap.nelem, 25)

        # make sure we got a file
        self.assertTrue(os.path.exists(self.testndf),
                        "Test existence of NDF file")

    def test_newwithwrite(self):
        # okay we have all the data, time to open us up an ndf
        indf = ndf.open(self.testndf, 'WRITE', 'NEW')
        indf.new('_REAL', 2,
                 np.array([0, 0]), np.array([4, 4]))

        # create PAMELA extension
        loc = indf.xnew('PAMELA', 'STRUCT')

        hdsloc = hds._transfer(loc)
        name = hdsloc.name
        self.assertEqual(name, "PAMELA")

        ccd = np.zeros([5, 5])

        # map primary data
        ndfmap = indf.map('DATA', '_REAL', 'WRITE')
        self.assertEqual(ndfmap.type, "_REAL")
        ndfmap.numpytondf(ccd)

        # Attribute testing
        indf.title = "A Title"
        self.assertEqual(indf.title, "A Title")
        self.assertIsNone(indf.units)
        indf.units = "K"
        self.assertEqual(indf.units, "K")
        indf.units = None
        self.assertIsNone(indf.units)

        # shut down ndf system
        indf.annul()

        # make sure we got a file
        self.assertTrue(os.path.exists(self.testndf),
                        "Test existence of NDF file")


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
