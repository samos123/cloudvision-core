from __future__ import print_function
import io
import sys
import os

import cv2
import numpy as np


def extract_opencv_features(feature_name):

    def extract_opencv_features_nested(imgfile_imgbytes):
        imgfilename, imgbytes = imgfile_imgbytes
        nparr = np.fromstring(buffer(imgbytes), np.uint8)
        img = cv2.imdecode(nparr, 1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if feature_name in ["surf", "SURF"]:
            extractor = cv2.xfeatures2d.SURF_create()
        elif feature_name in ["sift", "SIFT"]:
            extractor = cv2.xfeatures2d.SIFT_create()

        kp, descriptors = extractor.detectAndCompute(gray, None)

        return (imgfilename, descriptors)

    return extract_opencv_features_nested


def serialize_numpy_array(numpy_array):
    output = io.BytesIO()
    np.savez_compressed(output, x=numpy_array)
    return output.getvalue()


def deserialize_numpy_array(savez_data):
    data = np.load(io.BytesIO(savez_data))
    return data["x"]


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

    images_surf = images.map(extract_opencv_features(feature_name))
    images_surf = images_surf.map(lambda x: (x[0], serialize_numpy_array(x[1])))\
                             .saveAsSequenceFile(feature_sequencefile_path)

