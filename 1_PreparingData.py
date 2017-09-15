import numpy as np
import os

if __name__ == '__main__':
	trainDataDir='caviar_data/case2/train1/'
	testDataDir='cavaiar_data/case2/test1/'

	# For training
	trainTextFile =open('data/train.txt', "w")
	for root, dirs, files in os.walk(trainDataDir):
		for d in dirs:    
			personDir=os.path.join(root, d)
			for root1, dirs1, files1 in os.walk(personDir):
				for file in files1:			
					imgFile='%s/%s' % (d,file)
					print (imgFile)
					trainTextFile.write('%s %d\n' % (imgFile,int(d)-1))
	trainTextFile.close()

	# For testing
	trainTextFile =open('data/test.txt', "w")
	for root, dirs, files in os.walk(testDataDir):
		for d in dirs:    
			personDir=os.path.join(root, d)
			for root1, dirs1, files1 in os.walk(personDir):
				for file in files1:			
					imgFile='%s/%s' % (d,file)
					print (imgFile)
					trainTextFile.write('%s %d\n' % (imgFile,int(d)-1))
	trainTextFile.close()