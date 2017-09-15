# Person Re-identification: training, testing and evaluation with Caffe  
![smp img](https://raw.githubusercontent.com/zzvvmm/reid-caffe/master/caviar_data/sample.PNG)  

## Preparation
Edit the local of Caffe and dataset folder on file 1_, 2_, 6_, 7_...
## Operation
Run these files in order 1_, 2_, 3_, 4_, 5_, 6_, 7_(optinal)...  
1_Preparing training set and testing set  
2_Making lmdb database  
3_Making mean values  
4_Convert mean. binaryproto to mean.py (option)  
5_Fine-tuning or training  
6_Testing or classifying (option)  
7_Extracting feature (option)  

## Results on CAVIAR4REID dataset with GoogleNet CNN
CAVIAR4REID is a dataset for evaluating person re-identification algorithms. As the name suggest, the dataset has been extracted from the CAVIAR dataset mostly famous for person tracking and detection evaluations.  
![cnn img](https://raw.githubusercontent.com/zzvvmm/reid-caffe/master/caviar_data/CNN.PNG)  
The results are generally around 80% on rank-1.  
