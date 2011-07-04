import unittest
from starlink.ndf.Ndf import Ndf
import os.path

class TestFullRead(unittest.TestCase):

    def setUp(self):
        # Read the NDF for each test
        self.ndf = Ndf( os.path.join('data','ndf_test.sdf') )

    def test_label(self):
        self.assertEqual( self.ndf.label, 'Signal' )

    def test_title(self):
        self.assertEqual( self.ndf.title, 'Test Data' )

    def test_units(self):
        self.assertEqual( self.ndf.units, 'counts' )

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


