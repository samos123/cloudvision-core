package org.samos.cloudvision

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql._
import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.functions.udf


object KMeansDictionary {


  def main(args: Array[String]) {
    val k = args(0).toInt
    val featurePath = args(1)
    val kMeansModelPath = args(2)

    val conf = new SparkConf().setAppName("Scala KMeans Dictionary Generation")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val featuresDF = sqlContext.read.parquet(featurePath)
    // Flatten the Array[Array[Double]] to Array[Double] x * 128 becomes 1 * 128
    val featuresRDD = featuresDF.flatMap(x => x.getAs[Seq[Seq[Double]]]("features"))
    val featuresVectorRDD = featuresRDD.map(x => Vectors.dense(x.toArray)).cache()
    val model = KMeans.train(featuresVectorRDD, k, 5, 1, "random")

    model.save(sc, kMeansModelPath)

  }
}
