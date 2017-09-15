# Make sure that caffe is on the python path:
caffe_root = '../../caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np

#Convert mean file produced by Caffe to numpy array, assume 3 chanels
#python 4_convert_mean_to_npy.py 224 224 data/caviar_mean.binaryproto data/caviar_mean.npy

channels = 3

a = caffe.io.caffe_pb2.BlobProto()
with open(sys.argv[3],'rb') as f:
  a.ParseFromString(f.read())

means=a.data
means=np.asarray(means)
print means.shape
h = int(sys.argv[1])
w = int(sys.argv[2])
means=means.reshape(channels,h,w)
np.save(sys.argv[4],means)