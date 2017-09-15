##### Statistic dataset: all images of all persons in a folder 
import numpy as np
import os

if __name__ == '__main__':
	dataSetDir='/home/nhquan/works/datasets/CAVIAR/'
	maxPersonId=72
	arr=np.zeros(maxPersonId)

	totalImage=0
	avg=0

	# For training
	outputFile =open('data/statistic.txt', "w")
	for root, dirs, files in os.walk(dataSetDir):
		for file in files:					
			personId=int(file[:4])				
			arr[personId-1] +=1
	for i in range(maxPersonId):
		print ('%4d \t %d\n' % (i+1,arr[i]))
		outputFile.write('%4d \t %d\n' % (i+1,arr[i]))
		totalImage +=arr[i]	
	avg=totalImage/maxPersonId
	print ('Total: %d images' % (totalImage))
	outputFile.write('Total: %d images\n' % (totalImage))
	print ('Average: %.3f images per id' % (avg))
	outputFile.write('Average: %.3f images per id\n' % (avg))
	outputFile.close()

	