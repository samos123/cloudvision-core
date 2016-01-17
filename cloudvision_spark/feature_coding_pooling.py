from __future__ import print_function
import functools
import io
import sys
import os

import numpy as np
from scipy.spatial import distance
from pyspark.mllib.clustering import KMeansModel

from utils import serialize_numpy_array, deserialize_numpy_array

SUPPORTED_POOLING = ["max", "sum"]

def assign_pooling(img_feature_matrix, clusterCenters, pooling):
    image_name, feature_matrix = img_feature_matrix
    clusterCenters = clusterCenters.value
    model = KMeansModel(clusterCenters)
    bow = np.zeros(len(clusterCenters))

    for x in feature_matrix:
        k = model.predict(x)
        dist = distance.euclidean(clusterCenters[k], x)
        if pooling == "max":
            bow[k] = max(bow[k], dist)
        elif pooling == "sum":
            bow[k] = bow[k] + dist

    return (image_name, bow)


if __name__ == "__main__":
    from pyspark import SparkContext
    sc = SparkContext(appName="kmeans_assign")

    try:
        feature_sequencefile_path = sys.argv[1]
        kmeans_model_path = sys.argv[2]
        bow_sequencefile_path = sys.argv[3]
        pooling = sys.argv[4]
        partitions = int(sys.argv[5])

    except:
        print("Usage: spark-submit feature_coding_pooling.py "
              "<feature_sequencefile_path> <kmeans_model> "
              "<bow_sequencefile_path> <pooling_method:max> <partitions>")

    if pooling not in SUPPORTED_POOLING:
        raise ValueError("Pooling method %s is not supported. Supported poolings methods: %s" % (pooling, SUPPORTED_POOLING))

    features = sc.pickleFile(feature_sequencefile_path).repartition(partitions)
    model = KMeansModel.load(sc, kmeans_model_path)
    clusterCenters = model.clusterCenters
    clusterCenters = sc.broadcast(clusterCenters)

    features = features.map(lambda x: (x[0], deserialize_numpy_array(x[1])))
    features_bow = features.map(functools.partial(assign_pooling,
        clusterCenters=clusterCenters, pooling=pooling))

    features_bow = features_bow.map(lambda x: (x[0], serialize_numpy_array(x[1])))
    features_bow.saveAsPickleFile(bow_sequencefile_path)
    sc.stop()
