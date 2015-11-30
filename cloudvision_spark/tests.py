from __future__ import print_function
import os
import unittest
import sys

import numpy as np
import numpy.random


import feature_extraction
import kmeans_assign

class FeatureExtractionTest(unittest.TestCase):

    def test_extract_opencv_features(self):
        dir = os.path.dirname(__file__)
        imgpath = os.path.join(dir, 'img.jpg')
        imgbytes = bytearray(open(imgpath, "rb").read())
        result = feature_extraction.extract_opencv_features("surf")(("testfilename.jpg", imgbytes))
        self.assertEqual(result[0], "testfilename.jpg")

        result = feature_extraction.extract_opencv_features("sift")(("testfilename.jpg", imgbytes))
        self.assertEqual(result[0], "testfilename.jpg")


class KMeansAssignTests(unittest.TestCase):

    def _test_asign(self, pooling):
        img_feature_matrix = ("test.jpg", numpy.random.rand(500, 128))
        clusterCenters = numpy.random.rand(400, 128)
        clusterCenters = type('Dummy', (object,), { "value": clusterCenters })
        features_bow = kmeans_assign.assign_pooling(img_feature_matrix, clusterCenters=clusterCenters, pooling=pooling)

        self.assertEquals(features_bow[0], "test.jpg")
        self.assertEquals(features_bow[1].size, 400)

    def test_assign_max(self):
        self._test_asign("max")

    def test_assign_max(self):
        self._test_asign("sum")



if __name__ == "__main__":
    unittest.main()

