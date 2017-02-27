import sys
import os
import numpy as np
import VNet as VN

basePath=os.getcwd()

params = dict()
params['DataManagerParams']=dict()
params['ModelParams']=dict()

#params of the algorithm
params['ModelParams']['numcontrolpoints']=2
params['ModelParams']['sigma']=15
params['ModelParams']['device']=0

params['ModelParams']['prototxtTrain']=os.path.join(basePath,'Prototxt/train_noPooling_ResNet_cinque_target.prototxt')
params['ModelParams']['prototxtTest']=os.path.join(basePath,'Prototxt/test_noPooling_ResNet_cinque_target.prototxt')
params['ModelParams']['snapshot']=0
#params['ModelParams']['snapshot']=12000
params['ModelParams']['dirTrain']=os.path.join(basePath,'Dataset/data')
params['ModelParams']['dirTest']=os.path.join(basePath,'Dataset/Test/V16609')
params['ModelParams']['dirResult']=os.path.join(basePath,'Results') #where we need to save the results (relative to the base path)
params['ModelParams']['dirSnapshots']=os.path.join(basePath,'Models/') #where to save the models while training
params['ModelParams']['modelPath']=os.path.join(basePath,'Models/ctv/_iter_12000.caffemodel')
params['ModelParams']['batchsize'] = 2 #the batchsize
params['ModelParams']['numIterations'] = 100000 #the number of iterations
#params['ModelParams']['baseLR'] = 0.0001 #the learning rate, initial one
params['ModelParams']['baseLR'] = 0.0001 #the learning rate, initial one
params['ModelParams']['nProc'] = 4 #the number of threads to do data augmentation


#params of the DataManager
#params['DataManagerParams']['labelList'] = ["CTV"]
params['DataManagerParams']['labelList'] = ["PTV"]
params['DataManagerParams']['dstRes'] = np.asarray([1.8,1.8,5.0],dtype=float)
#params['DataManagerParams']['NumVolSize'] = np.asarray([128,128,64],dtype=int)
params['DataManagerParams']['NumVolSize'] = np.asarray([192,192,64],dtype=int)
params['DataManagerParams']['VolSize'] = np.asarray([128,128,16],dtype=int)
params['DataManagerParams']['normDir'] = False #if rotates the volume according to its transformation in the mhd file. Not reccommended.

'''
if __name__ == "__main__":
    source_path = sys.argv[1]
    dest_path = sys.argv[2]
    #dicom_path = sys.argv[3]
    #params['ModelParams']['dirTest'] = dicom_path
    model=VN.VNet(params)
    #model.train()
    model.test(source_path,dest_path)
'''

if __name__ == "__main__":
    model=VN.VNet(params)
    model.train()

'''
model=VN.VNet(params)
train = [i for i, j in enumerate(sys.argv) if j == '-train']
if len(train)>0:
    model.train()

test = [i for i, j in enumerate(sys.argv) if j == '-test']
if len(test) > 0:
    model.test()
'''


