#!/bin/bash

k=50000
maxiter=100
partitions=300


spark-submit --conf spark.driver.maxResultSize=0 --conf spark.mesos.coarse=true --executor-memory 5g --driver-memory 5g /tmp/SamKmeans-1.2.jar /tmp/5M.txt 50000 100 1000
