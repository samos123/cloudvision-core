from __future__ import print_function
import os
import unittest

import numpy as np

import feature_extraction

class FeatureExtractionTest(unittest.TestCase):

    def test_extract_opencv_features(self):
        dir = os.path.dirname(__file__)
        imgpath = os.path.join(dir, 'img.jpg')
        imgbytes = bytearray(open(imgpath, "rb").read())
        result = feature_extraction.extract_opencv_features("surf")(("testfilename.jpg", imgbytes))
        self.assertEqual(result[0], "testfilename.jpg")

        result = feature_extraction.extract_opencv_features("sift")(("testfilename.jpg", imgbytes))
        self.assertEqual(result[0], "testfilename.jpg")




if __name__ == "__main__":
    unittest.main()

