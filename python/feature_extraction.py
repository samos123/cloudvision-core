from __future__ import print_function
import sys
import os

import cv2
import numpy as np


def extract_surf_features_opencv(imgfile_imgbytes):
    imgfilename, imgbytes = imgfile_imgbytes
    nparr = np.fromstring(buffer(imgbytes), np.uint8)
    img = cv2.imdecode(nparr, 1)
#
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
#
    kp, descriptors = surf.detectAndCompute(gray, None)
   
    return (imgfilename, descriptors)


if __name__ == "__main__":
    from pyspark import SparkContext
    sc = SparkContext(appName="feature_extractor")

    try:
        image_seqfile_path = sys.argv[1]
        feature_sequencefile_path = sys.argv[2]
    except:
        print("Usage: spark-submit feature_extraction.py "
              "<image_sequencefile_input_path> <feature_sequencefile_output_path>")

    images = sc.sequenceFile(image_seqfile_path)

    images_surf = images.map(extract_surf_features_opencv)
    images_surf = images_surf.map(lambda x: (x[0], x[1].tostring()))\
                             .saveAsSequenceFile(feature_sequencefile_path)

