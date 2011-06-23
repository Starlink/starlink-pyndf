import unittest
import starlink.ndf as ndf
import os.path

class TestFullRead(unittest.TestCase):

    def setUp(self):
        # Read the NDF for each test
        self.ndf = ndf.Ndf( os.path.join('data','ndf_test.sdf') )

    def test_label(self):
        self.assertEqual( self.ndf.label, 'Signal' )

    def test_title(self):
        self.assertEqual( self.ndf.title, 'Test Data' )

    def test_units(self):
        self.assertEqual( self.ndf.units, 'counts' )
