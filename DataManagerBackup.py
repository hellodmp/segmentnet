import numpy as np
import SimpleITK as sitk
from LabelManager import LabelManager
from os import listdir
from os.path import isfile, isdir, join, splitext

class DataManager(object):
    params=None
    srcFolder=None
    resultsDir=None

    fileList=None
    gtList=None

    sitkImages=None
    sitkGT=None
    meanIntensityTrain = None
    label_list = ["Urinary Bladder","FemoralHead"]

    def __init__(self, srcFolder, resultsDir, parameters):
        self.params=parameters
        self.srcFolder=srcFolder
        self.resultsDir=resultsDir

    def createImageFileList(self):
        self.fileList = [f for f in listdir(self.srcFolder) if isdir(join(self.srcFolder, f))]
        print 'FILE LIST: ' + str(self.fileList)


    def loadImages(self):
        self.sitkImages = dict()
        rescalFilt = sitk.RescaleIntensityImageFilter()
        rescalFilt.SetOutputMaximum(1)
        rescalFilt.SetOutputMinimum(0)
        stats = sitk.StatisticsImageFilter()
        reader = sitk.ImageSeriesReader()
        m = 0.0
        for dir in self.fileList:
            dir = join(self.srcFolder, dir)
            series_list = reader.GetGDCMSeriesIDs(dir)
            for series_id in series_list:
                dicom_names = reader.GetGDCMSeriesFileNames(dir, series_id)
                if len(dicom_names) > 1:
                    break
            reader.SetFileNames(dicom_names)
            self.sitkImages[dir] = rescalFilt.Execute(sitk.Cast(reader.Execute(),sitk.sitkFloat32))
            stats.Execute(self.sitkImages[dir])
            m += stats.GetMean()
        self.meanIntensityTrain = m / len(self.sitkImages)


    def loadTrainingData(self):
        self.createImageFileList()
        self.loadImages()
        #load labels
        key = self.sitkImages.keys()[0]
        spacing = self.sitkImages[key].GetSpacing()
        manager = LabelManager(self.srcFolder, spacing)
        manager.createLabelFileList()
        self.sitkGT = manager.load_labels(self.label_list)
        #self.createGTFileList()
        #self.loadGT()

    def loadTestData(self):
        self.createImageFileList()
        self.loadImages()

    def getNumpyImages(self):
        dat = self.getNumpyData(self.sitkImages,sitk.sitkLinear)
        return dat


    def getNumpyGT(self):
        dat = self.getNumpyData(self.sitkGT,sitk.sitkLinear)

        for key in dat:
            dat[key] = (dat[key]>0.5).astype(dtype=np.float32)

        return dat


    def getNumpyData(self, dat, method):
        ret=dict()
        for key in dat:
            ret[key] = np.zeros([self.params['NumVolSize'][0], self.params['NumVolSize'][1], self.params['NumVolSize'][2]], dtype=np.float32)

            img=dat[key]
            #print "image_spacing=",img.GetSpacing()
            #print "image_size=", img.GetSize()

            #we rotate the image according to its transformation using the direction and according to the final spacing we want
            factor = np.asarray(img.GetSpacing()) / [self.params['dstRes'][0], self.params['dstRes'][1],
                                                     self.params['dstRes'][2]]

            factorSize = np.asarray(img.GetSize() * factor, dtype=float)

            newSize = np.max([factorSize, self.params['NumVolSize']], axis=0)

            newSize = newSize.astype(dtype=int)

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            resampler.SetOutputSpacing([self.params['dstRes'][0], self.params['dstRes'][1], self.params['dstRes'][2]])
            resampler.SetSize(newSize)
            resampler.SetInterpolator(method)
            if self.params['normDir']:
                T=sitk.AffineTransform(3)
                T.SetMatrix(img.GetDirection())
                resampler.SetTransform(T.GetInverse())

            imgResampled = resampler.Execute(img)
            imgCentroid = np.asarray(newSize, dtype=float) / 2.0
            imgStartPx = (imgCentroid - self.params['NumVolSize'] / 2.0).astype(dtype=int)
            regionExtractor = sitk.RegionOfInterestImageFilter()
            regionExtractor.SetSize(list(self.params['NumVolSize'].astype(dtype=int)))
            regionExtractor.SetIndex(list(imgStartPx))

            imgResampledCropped = regionExtractor.Execute(imgResampled)

            ret[key] = np.transpose(sitk.GetArrayFromImage(imgResampledCropped).astype(dtype=float), [1, 2, 0])
        return ret


    def writeResultsFromNumpyLabel(self,result,key):
        img = self.sitkImages[key]

        toWrite=sitk.Image(img.GetSize()[0],img.GetSize()[1],img.GetSize()[2],sitk.sitkFloat32)

        factor = np.asarray(img.GetSpacing()) / [self.params['dstRes'][0], self.params['dstRes'][1],
                                                     self.params['dstRes'][2]]

        factorSize = np.asarray(img.GetSize() * factor, dtype=float)

        newSize = np.max([factorSize, self.params['VolSize']], axis=0)

        newSize = newSize.astype(dtype=int)

        T = sitk.AffineTransform(3)
        T.SetMatrix(img.GetDirection())

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        resampler.SetOutputSpacing([self.params['dstRes'][0], self.params['dstRes'][1], self.params['dstRes'][2]])
        resampler.SetSize(newSize)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        print "start transfrom"

        if self.params['normDir']:
            resampler.SetTransform(T.GetInverse())

        toWrite = resampler.Execute(toWrite)

        imgCentroid = np.asarray(newSize, dtype=float) / 2.0

        imgStartPx = (imgCentroid - self.params['VolSize'] / 2.0).astype(dtype=int)

        for dstX, srcX in zip(range(0, result.shape[0]), range(imgStartPx[0],int(imgStartPx[0]+self.params['VolSize'][0]))):
            for dstY, srcY in zip(range(0, result.shape[1]), range(imgStartPx[1], int(imgStartPx[1]+self.params['VolSize'][1]))):
                for dstZ, srcZ in zip(range(0, result.shape[2]), range(imgStartPx[2], int(imgStartPx[2]+self.params['VolSize'][2]))):
                    try:
                        toWrite.SetPixel(int(srcX),int(srcY),int(srcZ),float(result[dstX,dstY,dstZ]))
                    except:
                        pass


        resampler.SetOutputSpacing([img.GetSpacing()[0], img.GetSpacing()[1], img.GetSpacing()[2]])
        resampler.SetSize(img.GetSize())

        if self.params['normDir']:
            resampler.SetTransform(T)

        toWrite = resampler.Execute(toWrite)

        thfilter=sitk.BinaryThresholdImageFilter()
        thfilter.SetInsideValue(1)
        thfilter.SetOutsideValue(0)
        thfilter.SetLowerThreshold(0.5)
        toWrite = thfilter.Execute(toWrite)

        #connected component analysis (better safe than sorry)

        cc = sitk.ConnectedComponentImageFilter()
        toWritecc = cc.Execute(sitk.Cast(toWrite,sitk.sitkUInt8))

        arrCC=np.transpose(sitk.GetArrayFromImage(toWritecc).astype(dtype=float), [2, 1, 0])

        lab=np.zeros(int(np.max(arrCC)+1),dtype=float)

        for i in range(1,int(np.max(arrCC)+1)):
            lab[i]=np.sum(arrCC==i)

        activeLab=np.argmax(lab)

        toWrite = (toWritecc==activeLab)

        toWrite = sitk.Cast(toWrite,sitk.sitkUInt8)

        writer = sitk.ImageFileWriter()
        #filename, ext = splitext(key)
        #print join(self.resultsDir, filename + '_result' + ext)
        #writer.SetFileName(join(self.resultsDir, filename + '_result' + ext))
        filename = key+".raw"
        print filename
        writer.SetFileName(filename)
        writer.Execute(toWrite)

