from __future__ import print_function
import io
import sys
import os

import numpy as np
from pyspark.mllib.clustering import KMeans

from utils import serialize_numpy_array, deserialize_numpy_array


if __name__ == "__main__":
    from pyspark import SparkContext
    sc = SparkContext(appName="kmeans_dictionary_creation")

    try:
        k = sys.argv[1]
        feature_sequencefile_path = sys.argv[2]
        kmeans_model_path = sys.argv[3]
    except:
        print("Usage: spark-submit kmeans.py <k:clusters> "
              "<feature_sequencefile_input_path> <kmeans_model_output>")

    features = sc.pickleFile(feature_sequencefile_path)

    features = features.map(lambda x: deserialize_numpy_array(x[1]))
    features = features.flatMap(lambda x: x.tolist())
    model = KMeans.train(features, int(k))
    sc.parallelize(model.clusterCenters).saveAsPickleFile(kmeans_model_path)
    print("Clusters have been saved as text file to %s" % kmeans_model_path)
    print("Final centers: " + str(model.clusterCenters))
    sc.stop()
