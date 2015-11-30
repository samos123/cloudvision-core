#!/bin/bash

k=50000
maxiter=100
partitions=300

# Previous java job
spark-submit --conf spark.driver.maxResultSize=0 --conf spark.mesos.coarse=true --executor-memory 5g --driver-memory 5g /tmp/SamKmeans-1.2.jar /tmp/5M.txt 50000 100 1000

# Python jobs
spark-submit --py-files utils.py feature_extraction.py sift tachyon://mesos-master:19999/seq-file tachyon://mesos-master:19999/feature-pickle-file
