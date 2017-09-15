#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

CAFFE=../../caffe
DATA=data
TOOLS=$CAFFE/build/tools

$TOOLS/compute_image_mean $DATA/caviar_train_lmdb \
  $DATA/caviar_mean.binaryproto

echo "Done."
