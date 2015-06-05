# Cloudvision - Computer Vision Cloud Platform

On demand large scale computer vision at your finger tops. Cloudvision
is meant to provide the computer vision industry with a scalable
platform to solve large scale computer vision problems.


    +----------+ +------------+  +----------------+
    |Feature   | |Quantization|  |Machine learning|
    |extraction| +------------+  +----------------+
    +----------+                                   
             on top of                                                
             +-----+ +-----+                       
             |HDFS | |Spark|                       
             +-----+ +-----+                       

## Setup development environment in a VM

This repo contains a Vagrantfile with Ansible scripts to
automatically deploy HDFS and Spark in standalone mode.

Execute the following commands to setup this VM:

    ./scripts/setup.sh
    vagrant ssh

