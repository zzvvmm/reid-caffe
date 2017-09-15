# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import os, random
import sys
import cv2
sys.path.insert(0, '/home/binhnt/works/caffe-projects/caviar2/hihi/hihi/getch/')
import getch
import time

# display plots in this notebook
#%matplotlib inline


def main(argv):

    project_dir='./'
    image_dir=argv[1]
        
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
    # model_weights='/home/binhnt/works/caffe-projects/bvlc_googlenet.caffemodel'

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
    
    image = caffe.io.load_image(image_dir)
    transformed_image = transformer.preprocess('data', image)        
    net.blobs['data'].data[...] = transformed_image
    ### perform classification
    output = net.forward()
    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch          
    top_inds = output_prob.argsort()[::-1][:5]   
    i=0
    while i<6:
        print
        print
        print 'Anh nay la cua nguoi: ', top_inds[i]+1
        print 'Neu dung thi an y de xac nhan! y?'
        char = getch.getch()
        if char=='y':                 
            #cv2.imshow('Anh TEST', image)
            fig = plt.figure()
            plt.imshow(image)
            fig.suptitle('Anh Test')
            plt.get_current_fig_manager().window.setGeometry(200,200,256,256)
            #cv2.moveWindow("Anh TEST", 0, 0) 
            if top_inds[i]>8:
                img_random=random.choice(os.listdir('/home/binhnt/works/CAVIAR4REID/train/' + '00' + str(top_inds[i]+1)))
                img_rddir='/home/binhnt/works/CAVIAR4REID/train/' + '00' + str(top_inds[i]+1) +'/' + img_random
            else:
                img_random=random.choice(os.listdir('/home/binhnt/works/CAVIAR4REID/train/' + '000' + str(top_inds[i]+1)))
                img_rddir='/home/binhnt/works/CAVIAR4REID/train/' + '000' + str(top_inds[i]+1) +'/' + img_random

            img_same = cv2.imread(img_rddir)       
            fig2=plt.figure()
            fig2.suptitle('Anh Tim Thay')
            plt.imshow(img_same)    
            plt.get_current_fig_manager().window.setGeometry(456,200,256,256)
            #cv2.imshow('Anh Tim Duoc', img_same)
            #cv2.moveWindow("Anh Tim Duoc", 400, 0)            
            #key = cv2.waitKey(5000)
            plt.show()
            #cv2.destroyAllWindows()
            break
        i+=1
    
    del net
if __name__ == '__main__':    
    main(sys.argv)
