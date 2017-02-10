import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import DataManager as DM
import utilities
from os.path import splitext
from multiprocessing import Process, Queue

class VNet(object):
    params=None
    dataManagerTrain=None
    dataManagerTest=None

    def __init__(self,params):
        self.params=params
        caffe.set_device(self.params['ModelParams']['device'])
        caffe.set_mode_gpu()
        #caffe.set_mode_cpu()

    def prepareDataThread(self, dataQueue, numpyImages, numpyGT):
        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']
        keysIMG = numpyImages.keys()

        nr_iter_dataAug = nr_iter * batchsize
        np.random.seed()
        whichDataList = np.random.randint(len(keysIMG), size=int(nr_iter_dataAug / self.params['ModelParams']['nProc']))

        for whichData in whichDataList:
            filename = keysIMG[whichData]
            defImg = numpyImages[filename]
            defLab = numpyGT[filename]
            (w,h,d) = defImg.shape
            flag = False
            for i in range(0,10):
                startw = np.random.randint(w - self.params['DataManagerParams']['VolSize'][0] + 1)
                starth = np.random.randint(h - self.params['DataManagerParams']['VolSize'][1] + 1)
                startd = np.random.randint(d - self.params['DataManagerParams']['VolSize'][2] + 1)

                defImg = defImg[startw:startw+self.params['DataManagerParams']['VolSize'][0],
                         starth:starth+self.params['DataManagerParams']['VolSize'][1],
                         startd:startd + self.params['DataManagerParams']['VolSize'][2]]

                defLab = defLab[:,startw:startw+self.params['DataManagerParams']['VolSize'][0],
                         starth:starth+self.params['DataManagerParams']['VolSize'][1],
                         startd:startd + self.params['DataManagerParams']['VolSize'][2]]
                if np.sum(defLab) > 50:
                    flag = True
                    break
            if not flag:
                continue
            dataQueue.put(tuple((defImg, defLab)))


    def trainThread(self,dataQueue,solver):

        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']

        batchData = np.zeros((batchsize, 1,
                              self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2]), dtype=float)
        batchLabel = np.zeros((batchsize, self.params['ModelParams']['labelsize'],
                               self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2]), dtype=float)

        train_loss = np.zeros(nr_iter)
        for it in range(nr_iter):
            for i in range(batchsize):
                [defImg, defLab] = dataQueue.get()

                batchData[i, 0, :, :, :] = defImg.astype(dtype=np.float32)
                batchLabel[i, :, :, :, :] = (defLab > 0.5).astype(dtype=np.float32)

            solver.net.blobs['data'].data[...] = batchData.astype(dtype=np.float32)
            solver.net.blobs['label'].data[...] = batchLabel.astype(dtype=np.float32)
            #solver.net.blobs['labelWeight'].data[...] = batchWeight.astype(dtype=np.float32)
            #use only if you do softmax with loss

            solver.step(1)  # this does the training
            train_loss[it] = solver.net.blobs['loss'].data

            if (np.mod(it, 10) == 0):
                plt.clf()
                plt.plot(range(0, it), train_loss[0:it])
                plt.pause(0.00000001)
            matplotlib.pyplot.show()



    def train(self):
        print self.params['ModelParams']['dirTrain']

        #we define here a data manage object
        self.dataManagerTrain = DM.DataManager(self.params['ModelParams']['dirTrain'],
                                               self.params['ModelParams']['dirResult'],
                                               self.params['DataManagerParams'])

        self.dataManagerTrain.loadTrainingData() #loads in sitk format

        howManyImages = len(self.dataManagerTrain.sitkImages)
        howManyGT = len(self.dataManagerTrain.sitkGT)

        assert howManyGT == howManyImages

        print "The dataset has shape: data - " + str(howManyImages) + ". labels - " + str(howManyGT)

        test_interval = 50000
        # Write a temporary solver text file because pycaffe is stupid
        with open("solver.prototxt", 'w') as f:
            f.write("net: \"" + self.params['ModelParams']['prototxtTrain'] + "\" \n")
            f.write("base_lr: " + str(self.params['ModelParams']['baseLR']) + " \n")
            f.write("momentum: 0.99 \n")
            f.write("weight_decay: 0.0005 \n")
            f.write("lr_policy: \"step\" \n")
            f.write("stepsize: 20000 \n")
            f.write("gamma: 0.1 \n")
            f.write("clip_gradients: 35 \n")
            f.write("display: 1 \n")
            f.write("snapshot: 2000 \n")
            f.write("snapshot_prefix: \"" + self.params['ModelParams']['dirSnapshots'] + "\" \n")
            #f.write("test_iter: 3 \n")
            #f.write("test_interval: " + str(test_interval) + "\n")

        f.close()

        solver = caffe.SGDSolver("solver.prototxt")
        os.remove("solver.prototxt")

        if (self.params['ModelParams']['snapshot'] > 0):
            solver.restore(self.params['ModelParams']['dirSnapshots'] + "_iter_" + str(
                self.params['ModelParams']['snapshot']) + ".solverstate")

        plt.ion()

        numpyImages = self.dataManagerTrain.getNumpyImages()
        numpyGT = self.dataManagerTrain.getNumpyGT()

        #numpyImages['Case00.mhd']
        #numpy images is a dictionary that you index in this way (with filenames)

        for key in numpyImages:
            mean = np.mean(numpyImages[key][numpyImages[key]>0])
            std = np.std(numpyImages[key][numpyImages[key]>0])
            numpyImages[key]-=mean
            numpyImages[key]/=std

        dataQueue = Queue(30) #max 50 images in queue
        dataPreparation = [None] * self.params['ModelParams']['nProc']

        #thread creation
        for proc in range(0,self.params['ModelParams']['nProc']):
            dataPreparation[proc] = Process(target=self.prepareDataThread, args=(dataQueue, numpyImages, numpyGT))
            dataPreparation[proc].daemon = True
            dataPreparation[proc].start()

        self.trainThread(dataQueue, solver)


    def filter(self, dat):
        (w,h,d) = dat.shape
        for i in range(0,d):
            count = np.sum(dat[:,:,i])
            if count < 40:
                dat[:, :, i] = np.zeros((w,h),dtype=float)
        return dat

    def dice(self, result, gt):
        union = (np.sum(result) + np.sum(gt))
        intersection = (np.sum(result * gt))
        dice_num = 2 * intersection / union
        print dice_num
        return dice_num


    def test(self):
        self.dataManagerTest = DM.DataManager(self.params['ModelParams']['dirTest'], self.params['ModelParams']['dirResult'], self.params['DataManagerParams'])
        self.dataManagerTest.loadTestData()

        net = caffe.Net(self.params['ModelParams']['prototxtTest'],
                        os.path.join(self.params['ModelParams']['dirSnapshots'],"_iter_" + str(self.params['ModelParams']['snapshot']) + ".caffemodel"),
                        caffe.TEST)

        numpyImages = self.dataManagerTest.getNumpyImages()
        numpyGT = self.dataManagerTest.getNumpyGT()
        numpyImages_back = self.dataManagerTest.getNumpyImages()
        total = 0

        for key in numpyImages:
            mean = np.mean(numpyImages[key][numpyImages[key]>0])
            std = np.std(numpyImages[key][numpyImages[key]>0])

            numpyImages[key] -= mean
            numpyImages[key] /= std

        for key in numpyImages:
            step = self.params['DataManagerParams']['VolSize'][2]
            result = np.zeros((self.params['DataManagerParams']['NumVolSize'][0],
                              self.params['DataManagerParams']['NumVolSize'][1],
                              self.params['DataManagerParams']['NumVolSize'][2]),dtype=float)
            #result = np.zeros(self.params['DataManagerParams']['NumVolSize'].shape,dtype=float)
            for i in range(self.params['DataManagerParams']['NumVolSize'][2]/step):
                #image_input = numpyImages[key][32:160, 32:160, i * step:(i + 1) * step]
                image_input = numpyImages[key][:, :, i * step:(i + 1) * step]
                btch = np.reshape(image_input, [1, 1, image_input.shape[0], image_input.shape[1], image_input.shape[2]])
                net.blobs['data'].data[...] = btch
                out = net.forward()
                l = out["labelmap"]
                result[:, :,i * step:(i + 1) * step] = np.squeeze(l[0, 0, :, :, :])
            result = self.filter(result)
            #self.dataManagerTest.writeResultsFromNumpyLabel(result, key)
            image = numpyImages_back[key]
            image[result >= 0.5] = 1
            #utilities.sitk_show(numpyGT[key][0])
            utilities.sitk_show(image)
            print key
            total += self.dice(numpyGT[key][0],result)
        print "total=", total
        print "count=", len(numpyImages)
        print "mean=", total/len(numpyImages)


    '''
    def test(self):
        self.dataManagerTest = DM.DataManager(self.params['ModelParams']['dirTest'], self.params['ModelParams']['dirResult'], self.params['DataManagerParams'])
        self.dataManagerTest.loadTestData()

        net = caffe.Net(self.params['ModelParams']['prototxtTest'],
                        os.path.join(self.params['ModelParams']['dirSnapshots'],"_iter_" + str(self.params['ModelParams']['snapshot']) + ".caffemodel"),
                        caffe.TEST)

        numpyImages = self.dataManagerTest.getNumpyImages()
        numpyImages_back = self.dataManagerTest.getNumpyImages()

        for key in numpyImages:
            mean = np.mean(numpyImages[key][numpyImages[key]>0])
            std = np.std(numpyImages[key][numpyImages[key]>0])

            numpyImages[key] -= mean
            numpyImages[key] /= std

        results = dict()

        for key in numpyImages:
            for i in range(0,2):
                image_input = numpyImages[key][32:160,32:160,i*16:(i+1)*16]
                btch = np.reshape(image_input,[1,1,image_input.shape[0],image_input.shape[1],image_input.shape[2]])

                net.blobs['data'].data[...] = btch

                out = net.forward()
                l = out["labelmap"]

                labelmap1 = np.squeeze(l[0,0,:,:,:])
                labelmap2 = np.squeeze(l[0, 1, :, :, :])

                #results[key] = np.squeeze(labelmap)

                #image = numpyImages_back[key]
                #image[results[key] > 0.5] = 1.0
                #utilities.sitk_show(image)

                image = numpyImages_back[key][32:160,32:160,i*16:(i+1)*16]
                result1 = np.squeeze(labelmap1)
                result2 = np.squeeze(labelmap2)
                utilities.sitk_show(image)
                image[result1 >= 0.5] = 1
                image[result2 >= 0.5] = 1
                utilities.sitk_show(image)
                self.dataManagerTest.writeResultsFromNumpyLabel(np.squeeze(l[0,0,:,:,:]),key)
    '''


