import unittest
import starlink.ndf.api as ndf
import os.path

class TestExceptions(unittest.TestCase):

    def test_filnf(self):
        with self.assertRaises(IOError):
            indf = ndf.open('shouldnotbepresent','READ','OLD')

    def test_badmode(self):
        with self.assertRaises(ValueError):
            indf = ndf.open('badmode', 'UNKNOWN', 'OLD')

    def test_badstate(self):
        with self.assertRaises(ValueError):
            indf = ndf.open('badmode', 'READ', 'VERYOLD')

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
