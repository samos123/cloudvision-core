from __future__ import print_function
import logging
import io
import sys
import os

import cv2
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row

import utils
from utils import serialize_numpy_array, deserialize_numpy_array, log, log_memory_usage


def extract_opencv_features(feature_name):

    def extract_opencv_features_nested(imgfile_imgbytes):
        try:
            imgfilename, imgbytes = imgfile_imgbytes
            nparr = np.fromstring(buffer(imgbytes), np.uint8)
            img = cv2.imdecode(nparr, 0)
            if feature_name in ["surf", "SURF"]:
                extractor = cv2.SURF()
            elif feature_name in ["sift", "SIFT"]:
                extractor = cv2.SIFT()

            kp, descriptors = extractor.detectAndCompute(img, None)

            log_memory_usage()

            return [(imgfilename, descriptors)]
        except Exception, e:
            logging.exception(e)
            return []

    return extract_opencv_features_nested


if __name__ == "__main__":
    sc = SparkContext(appName="feature_extractor")
    sqlContext = SQLContext(sc)

    try:
        feature_name = sys.argv[1]
        image_seqfile_path = sys.argv[2]
        feature_sequencefile_path = sys.argv[3]
        partitions = int(sys.argv[4])
    except:
        print("Usage: spark-submit feature_extraction.py <feature_name(sift or surf)> "
              "<image_sequencefile_input_path> <feature_sequencefile_output_path> <partitions>")

    images = sc.sequenceFile(image_seqfile_path, minSplits=partitions)

    features = images.flatMap(extract_opencv_features(feature_name))
    features = features.filter(lambda x: x[1] != None)
    features = features.map(lambda x: (Row(fileName=x[0], features=serialize_numpy_array(x[1]))))
    featuresSchema = sqlContext.createDataFrame(features)
    featuresSchema.registerTempTable("images")
    featuresSchema.write.parquet(feature_sequencefile_path)
