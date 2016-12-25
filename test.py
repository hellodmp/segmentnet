import numpy as np
import os

import DataManager as DM

basePath=os.getcwd()

params = dict()
params['DataManagerParams']=dict()
params['ModelParams']=dict()

#params of the algorithm
params['ModelParams']['numcontrolpoints']=2
params['ModelParams']['sigma']=15
params['ModelParams']['device']=0
params['ModelParams']['prototxtTrain']=os.path.join(basePath,'Prototxt/train_noPooling_ResNet_cinque.prototxt')
params['ModelParams']['prototxtTest']=os.path.join(basePath,'Prototxt/test_noPooling_ResNet_cinque.prototxt')
params['ModelParams']['snapshot']=0
params['ModelParams']['dirTrain']=os.path.join(basePath,'Dataset/Train1')
params['ModelParams']['dirTest']=os.path.join(basePath,'Dataset/Test')
params['ModelParams']['dirResult']=os.path.join(basePath,'Results') #where we need to save the results (relative to the base path)
params['ModelParams']['dirSnapshots']=os.path.join(basePath,'Models/MRI_cinque_snapshots/') #where to save the models while training
params['ModelParams']['batchsize'] = 2 #the batchsize
params['ModelParams']['numIterations'] = 100000 #the number of iterations
params['ModelParams']['baseLR'] = 0.0001 #the learning rate, initial one
params['ModelParams']['nProc'] = 1 #the number of threads to do data augmentation

#params of the DataManager
params['DataManagerParams']['dstRes'] = np.asarray([1,1,1.5],dtype=float)
params['DataManagerParams']['VolSize'] = np.asarray([128,128,64],dtype=int)
params['DataManagerParams']['normDir'] = False #if rotates the volume according to its transformation in the mhd file. Not reccommended.

print params['ModelParams']['dirTrain']

#we define here a data manage object
dataManagerTrain = DM.DataManager(params['ModelParams']['dirTrain'],
                                  params['ModelParams']['dirResult'],
                                  params['DataManagerParams'])

#dataManagerTrain.loadTrainingData() #loads in sitk format
dataManagerTrain.createImageFileList()
print "fileList=",dataManagerTrain.fileList
dataManagerTrain.createGTFileList()
print "gtlist=", dataManagerTrain.gtList
dataManagerTrain.loadImages()
dataManagerTrain.loadGT()

numpyImages = dataManagerTrain.getNumpyImages()
numpyGT = dataManagerTrain.getNumpyGT()
print numpyImages['Case00.mhd'].shape
print numpyGT['Case00_segmentation.mhd'].shape
print "hellodmp"

