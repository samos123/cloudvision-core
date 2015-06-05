from __future__ import print_function
import os
import unittest

import numpy as np

import feature_extraction

class FeatureExtractionTest(unittest.TestCase):

    def test_extract_opencv_surf(self):
        dir = os.path.dirname(__file__)
        imgpath = os.path.join(dir, 'img.jpg')
        imgbytes = bytearray(open(imgpath, "rb").read())
        result = feature_extraction.extract_surf_features_opencv(("testfilename.jpg", imgbytes))
        self.assertEqual(result[0], "testfilename.jpg")
        print(result)


if __name__ == "__main__":
    unittest.main()

