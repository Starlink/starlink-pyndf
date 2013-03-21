import unittest
import starlink.Ast
from starlink.ndfpack import Ndf
import os.path

class TestWcs(unittest.TestCase):

    def setUp(self):
        # Read the NDF for each test
        self.ndf = Ndf( os.path.join('data','ndf_test.sdf') )

    def test_gtwcs(self):
        self.assertIsInstance( self.ndf.wcs, starlink.Ast.FrameSet )
        self.assertEqual( self.ndf.wcs.Domain, "PIXEL" )

        NdfI = self.ndf
        GridFrame =  starlink.Ast.Frame(2,"Domain=PIXEL")
        BoxI = starlink.Ast.Box(GridFrame, 1, NdfI.bound[0][::-1], NdfI.bound[1][::-1])




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

if __name__ == "__main__":
    unittest.main()
