from __future__ import print_function
import functools
import io
import sys
import os

import numpy as np
from scipy.spatial import distance
from pyspark.mllib.clustering import KMeansModel
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row

SUPPORTED_POOLING = ["max", "sum"]

def assign_pooling(row, clusterCenters, pooling):
    image_name = row['fileName']
    feature_matrix = np.array(row['features'])
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

    return Row(fileName=image_name, bow=bow.tolist())


if __name__ == "__main__":
    sc = SparkContext(appName="kmeans_assign")
    sqlContext = SQLContext(sc)

    try:
        feature_parquet_path = sys.argv[1]
        kmeans_model_path = sys.argv[2]
        bow_parquet_path = sys.argv[3]
        pooling = sys.argv[4]

    except:
        print("Usage: spark-submit feature_coding_pooling.py "
              "<feature_sequencefile_path> <kmeans_model> "
              "<bow_sequencefile_path> <pooling_method:max>")

    if pooling not in SUPPORTED_POOLING:
        raise ValueError("Pooling method %s is not supported. Supported poolings methods: %s" % (pooling, SUPPORTED_POOLING))

    features = sqlContext.read.parquet(feature_parquet_path)
    model = KMeansModel.load(sc, kmeans_model_path)
    clusterCenters = model.clusterCenters
    clusterCenters = sc.broadcast(clusterCenters)

    features_bow = features.map(functools.partial(assign_pooling,
        clusterCenters=clusterCenters, pooling=pooling))
    featuresSchema = sqlContext.createDataFrame(features_bow)
    featuresSchema.registerTempTable("images")
    featuresSchema.write.parquet(bow_parquet_path)
    sc.stop()
