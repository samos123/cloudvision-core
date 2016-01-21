package org.samos.cloudvision

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql._
import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.functions.udf


object KMeansDictionary {


  def main(args: Array[String]) {
    val k = args(0).toInt
    val featurePath = args(1)
    val kMeansModelPath = args(2)
    val partitions = args(3).toInt

    val conf = new SparkConf().setAppName("Scala KMeans Dictionary Generation")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val features = sqlContext.read.parquet(featurePath)
    // Flatten the features into the array exploded
    val exploded = features.select(explode(features("features"))).toDF("features")
    val vectorSize = udf {(x: Vector) => x.size}
    val withSize = exploded.withColumn("size", vectorSize(exploded("features")))
    
    val kmeans = new KMeans()
      .setK(k)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMaxIter(2)
      .setInitMode("random")
    val model = kmeans.fit(exploded)
    model.save(kMeansModelPath)
  }
}
