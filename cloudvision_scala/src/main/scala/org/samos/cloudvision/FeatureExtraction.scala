package org.samos.cloudvision

import scala.collection.JavaConverters._

import org.apache.hadoop.io._
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors

import de.lmu.ifi.dbs.utilities.Arrays2
import de.lmu.ifi.dbs.jfeaturelib.features.SURF
import ij.process.ColorProcessor
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.File
import java.io.IOException
import java.net.URISyntaxException
import java.util.List
import javax.imageio.ImageIO




object FeatureExtraction {

  case class Image(name: String, features: Array[Vector])

  def extractFeatures(bytes: Array[Byte]): Array[Vector] = {
    val bufferedImage: BufferedImage = ImageIO.read(new ByteArrayInputStream(bytes))
    val image = new ColorProcessor(bufferedImage)
    val descriptor = new SURF()
    descriptor.run(image)
    // Returns java.util.List[Array[Double]] so convert to scala type
    descriptor.getFeatures().asScala.toArray.map(x => Vectors.dense(x))
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Scala Feature Extraction of SURF/SIFT")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    val featureName = args(0)
    val imageSeqfilePath = args(1)
    val featureSequencefilePath = args(2)
    val partitions = args(3).toInt
    println(f"Init with featurename: $featureName%s, seqFilePath: $imageSeqfilePath%s, featSeqFilePath: $featureSequencefilePath%s, partitions: $partitions%s")
    val images = sc.sequenceFile[String, BytesWritable](imageSeqfilePath, partitions)
    val features = images.mapPartitions{ iter =>
      iter.flatMap(x => {
          try {
            Some((x._1, extractFeatures(x._2.copyBytes)))
          } catch {
            case e: Exception => None
          }
        }
      )
    }
    //    val features = images.mapValues(_.copyBytes).mapValues(extractFeatures)

    val featuresDF = features.map(x => Image(x._1, x._2)).toDF()
    featuresDF.write.parquet(featureSequencefilePath)
  }
}
