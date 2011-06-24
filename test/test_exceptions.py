import unittest
import starlink.ndf as ndf
import os.path

class TestExceptions(unittest.TestCase):

    def test_filnf(self):
        with self.assertRaises(IOError):
            indf,place = ndf.ndf_open('shouldnotbepresent','READ','OLD')

    def test_badmode(self):
        with self.assertRaises(ValueError):
            indf,place = ndf.ndf_open('badmode', 'UNKNOWN', 'OLD')

    def test_badstate(self):
        with self.assertRaises(ValueError):
            indf,place = ndf.ndf_open('badmode', 'READ', 'VERYOLD')
