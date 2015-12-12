from __future__ import print_function
import logging
import io
import sys
import os

import cv2
import numpy as np

from utils import serialize_numpy_array, deserialize_numpy_array


def extract_opencv_features(feature_name):

    def extract_opencv_features_nested(imgfile_imgbytes):
        try:
            imgfilename, imgbytes = imgfile_imgbytes
            nparr = np.fromstring(buffer(imgbytes), np.uint8)
            img = cv2.imdecode(nparr, 0)
            if feature_name in ["surf", "SURF"]:
                extractor = cv2.xfeatures2d.SURF_create()
            elif feature_name in ["sift", "SIFT"]:
                extractor = cv2.xfeatures2d.SIFT_create()

            kp, descriptors = extractor.detectAndCompute(img, None)

            return [(imgfilename, descriptors)]
        except Exception, e:
            logging.exception(e)
            return []

    return extract_opencv_features_nested


if __name__ == "__main__":
    from pyspark import SparkContext
    sc = SparkContext(appName="feature_extractor")

    try:
        feature_name = sys.argv[1]
        image_seqfile_path = sys.argv[2]
        feature_sequencefile_path = sys.argv[3]
    except:
        print("Usage: spark-submit feature_extraction.py <feature_name(sift or surf)> "
              "<image_sequencefile_input_path> <feature_sequencefile_output_path>")

    images = sc.sequenceFile(image_seqfile_path)

    features = images.flatMap(extract_opencv_features(feature_name))
    features = features.map(lambda x: (x[0], serialize_numpy_array(x[1])))
    features.saveAsPickleFile(feature_sequencefile_path)

