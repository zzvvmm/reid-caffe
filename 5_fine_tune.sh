#!/usr/bin/env sh
set -e
../../caffe/build/tools/caffe train \
  -solver models/googlenet/solver.prototxt \
  -weights ../bvlc_googlenet.caffemodel 2>&1 | tee -a log/train_caviar_googlenet.log
$@
