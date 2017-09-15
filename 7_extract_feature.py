# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy.io as sio
# display plots in this notebook
#%matplotlib inline

def main(argv):

	project_dir='./'
	test_data_dir='cavaiar_data/case2/test1'
	
	
	caffe_root = '../../caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
	sys.path.insert(0, caffe_root + 'python')
	import caffe
	# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

	#import os
	if os.path.isfile(project_dir + 'models/googlenet/bvlc_googlenet_iter_10000.caffemodel'):
		print 'model CaffeNet found.'
	else:
		print 'model CaffeNet not found.'
																																			
	#caffe.set_mode_cpu()
	caffe.set_device(0)  # if we have multiple GPUs, pick the first one
	caffe.set_mode_gpu()

	model_def = project_dir + 'models/googlenet/deploy.prototxt'
	model_weights = project_dir + 'models/googlenet/bvlc_googlenet_iter_10000.caffemodel'

	net = caffe.Net(model_def,      # defines the structure of the model
					model_weights,  # contains the trained weights
					caffe.TEST)     # use test mode (e.g., don't perform dropout)

	# load the mean ImageNet image (as distributed with Caffe) for subtraction
	mu = np.load(project_dir + 'data/caviar_mean.npy')
	mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
	print 'mean-subtracted values:', zip('BGR', mu)

	# create transformer for the input called 'data'
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
	transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

	# set the size of the input (we can skip this if we're happy
	#  with the default; we can also change it later, e.g., for different batch sizes)
	net.blobs['data'].reshape(1,        # batch size
							  3,         # 3-channel (BGR) images
							  224, 224)  # image size is 224x224
	img_dir=test_data_dir
	test_file='data/train.txt'
	if not os.path.exists(test_file):
		print 'test file not found'    
	test_list = np.loadtxt(test_file, str, delimiter=' ')   
	count=1
	tmp_fea=np.zeros(1024)
	for item in test_list:
		count=count+1
		print 'Extracting feature:', item[0]
		image = caffe.io.load_image(test_data_dir + item[0])
		transformed_image = transformer.preprocess('data', image)        
		net.blobs['data'].data[...] = transformed_image				
		#output = net.forward(end='pool5/7x7_s1')	
		net.forward() # this will load the next mini-batch as defined in the net
		output = net.blobs['pool5/7x7_s1'].data # or whatever you want
		
		# Save feature in spreate files
		matfile='features/' + item[0].split('/')[1] + '.mat';
		sio.savemat(matfile,{'output':output})

		# Save feature in one file
		tmp_fea = np.append(tmp_fea, output)
		tmp_fea=tmp_fea.reshape(count,1024)
		
	#sio.savemat('features/train_feat_py.mat',{'tmp_fea':tmp_fea[1:,:]})	

	
if __name__ == '__main__':    
	main(sys.argv)
