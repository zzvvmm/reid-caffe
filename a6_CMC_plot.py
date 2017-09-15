# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# display plots in this notebook
#%matplotlib inline

def main(argv):

	project_dir='/home/binhnt/works/caffe-projects/caviar2/'
	test_data_dir='/home/binhnt/work/Split_datasets/caviar/case1/test1/'
		
	# set display defaults
	#plt.rcParams['figure.figsize'] = (10, 10)        # large images
	#plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
	#plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

	# The caffe module needs to be on the Python path;
	#  we'll add it here explicitly.
	#import sys
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
	#model_weights='/home/nhquan/works/caffe_ethz/models/googlenet/bvlc_googlenet.caffemodel'

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
							  224, 224)  # image size is 227x227
	max_rank=20
	if (True): #not os.path.exists('output/CMC.npy'):	    
		CMC=np.zeros((max_rank+1), dtype=np.int)
		img_dir=test_data_dir
		test_file='data/test.txt'
		if not os.path.exists(test_file):
			print 'test file not found'    
		test_list = np.loadtxt(test_file, str, delimiter=' ')   
		count=0
		for item in test_list:
			count=count+1
			print 'Testing :', item[0]
			image = caffe.io.load_image(test_data_dir + item[0])
			transformed_image = transformer.preprocess('data', image)        
			net.blobs['data'].data[...] = transformed_image
			### perform classification
			output = net.forward()
			output_prob = output['prob'][0]  # the output probability vector for the first image in the batch          
			top_inds = output_prob.argsort()[::-1][:max_rank]   
			print 'top_inds:', top_inds
			# Display fail
			if int(item[1])!=top_inds[0]:     
				print 'Loss rank-1',int(item[1]), '<>', top_inds[0]
			for i in range(1,max_rank+1):    # 1.. max_rank				
				if int(item[1]) in top_inds[:i]:
					CMC[i]=CMC[i]+1
			print 'CMC: ', CMC
		CMC=(CMC/float(count))*100
		np.save('output/CMC.npy', CMC)
	else:
		CMC=np.load('output/CMC.npy')
	print 'CMC: ', CMC[1:]
	plt.plot(CMC)
	plt.axis([1, max_rank, 0, 100])
	plt.xlabel('Rank')
	plt.ylabel('Accuracy (%)')
	plt.title('Caffe - cavia (train: 72,val: 72)')
	plt.grid(True)
	plt.show()
	
if __name__ == '__main__':    
	main(sys.argv)
