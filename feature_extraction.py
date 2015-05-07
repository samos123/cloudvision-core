
import sys

import cv2
import numpy as np
from pyspark import SparkContext



def extract_sift_features_opencv(imgfilename, imgbytes):
    nparr = np.fromstring(imgbytes, np.uint8)
    imarr = cv2.imdecode(imgbytes,cv2.CV_LOAD_IMAGE_COLOR)

    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()

    kp, descriptors = sift.detectAndCompute(gray, None)
    
    return (imgfilename, descriptors)


if __name__ == "__main__":
    sc = SparkContext(appName="feature_extractor")

    seqFilePath = sys.args[1]
    images = sc.sequenceFile(seqFilePath)

    images_sift = images.map(extract_and_save_sift_features)
