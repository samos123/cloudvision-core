#!/bin/bash

mvn clean compile assembly:single

scp -i ~/.ssh/dinglab-417 target/*.jar ubuntu@mesos-master:/tmp/
