import unittest
import starlink.ndf as ndf
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

