/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package sam;

import java.util.regex.Pattern;
import java.util.UUID;
import java.io.BufferedWriter;
import java.io.OutputStreamWriter;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FSDataOutputStream;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

/**
 * Example using MLlib KMeans from Java.
 */
public final class JavaKMeans {

  private static class ParsePoint implements Function<String, Vector> {
    private static final Pattern SPACE = Pattern.compile(" ");

    @Override
    public Vector call(String line) {
      String[] tok = SPACE.split(line);
      double[] point = new double[tok.length];
      for (int i = 0; i < tok.length; ++i) {
        point[i] = Double.parseDouble(tok[i]);
      }
      return Vectors.dense(point);
    }
  }


  public static void main(String[] args) {
    if (args.length < 3) {
      System.err.println(
        "Usage: JavaKMeans <input_file> <k> <max_iterations> [<min_partitions>]");
      System.exit(1);
    }

    String inputFile = args[0];
    int k = Integer.parseInt(args[1]);
    int iterations = Integer.parseInt(args[2]);

    SparkConf sparkConf = new SparkConf().setAppName("JavaKMeans");
    JavaSparkContext sc = new JavaSparkContext(sparkConf);

    long startTime = System.currentTimeMillis();

    JavaRDD<String> lines;
    if (args.length >= 4) {
        int min_partitions = Integer.parseInt(args[3]);
        lines = sc.textFile(inputFile, min_partitions);
    } else {
        lines = sc.textFile(inputFile);
    }

    JavaRDD<Vector> points = lines.map(new ParsePoint());

    KMeansModel model = KMeans.train(points.rdd(), k, iterations, 1,  KMeans.K_MEANS_PARALLEL());

    Vector[] clusterCenters =  model.clusterCenters();

    String uuid = UUID.randomUUID().toString();

    // Write clusters to hdfs
    try {
        Configuration conf = new Configuration();
        conf.set("fs.default.name", "hdfs://mesos-master");
        FileSystem fs = FileSystem.get(conf);
        FSDataOutputStream out = fs.create(new Path("/tmp/k-means-result" + uuid), true);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(out));
        bw.write("test");
        int i = 0;
        for (Vector center : clusterCenters) {
          i = i + 1;
          bw.write("Cluster " + Integer.toString(i) + ": " + center + "\n");
        }
        out.close();

    } catch (Exception e) {
        e.printStackTrace();
    }


    System.out.println("Clustering result output has been written to hdfs://mesos-masters/tmp/k-means-result-" + uuid);

    System.out.println("Cluster centers:");
    for (Vector center :clusterCenters) {
      System.out.println(" " + center);
    }
    double cost = model.computeCost(points.rdd());
    System.out.println("Cost: " + cost);
    long elapsedTime = System.currentTimeMillis() - startTime;
    System.out.println("Elapsed time in minutes: " + (elapsedTime / 1000 / 60));


    sc.stop();
  }
}
