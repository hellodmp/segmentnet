import numpy as np
import SimpleITK as sitk
from LabelManager import LabelManager
from os import listdir
from os.path import isfile, isdir, join, splitext
import utilities
from RTExport import RTExport
from scipy import ndimage
from skimage import measure

class DataManager(object):
    params=None
    srcFolder=None
    resultsDir=None

    fileList=None
    gtList=None

    sitkImages=None
    sitkGT=None
    meanIntensityTrain = None
    #label_list = ["Urinary Bladder","FemoralHead"]
    #label_list = ["CTV","PTV"]

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
        reader = sitk.ImageSeriesReader()
        for dir in self.fileList:
            dir = join(self.srcFolder, dir)
            series_list = reader.GetGDCMSeriesIDs(dir)
            for series_id in series_list:
                dicom_names = reader.GetGDCMSeriesFileNames(dir, series_id)
                if len(dicom_names) > 1:
                    break
            reader.SetFileNames(dicom_names)
            self.sitkImages[dir] = [rescalFilt.Execute(sitk.Cast(reader.Execute(),sitk.sitkFloat32))]

    def loadTrainingData(self):
        self.createImageFileList()
        self.loadImages()
        #load labels
        key = self.sitkImages.keys()[0]
        spacing = self.sitkImages[key][0].GetSpacing()
        manager = LabelManager(self.srcFolder, spacing)
        manager.createLabelFileList()
        self.sitkGT = manager.load_labels(self.params['labelList'])

    def loadTestData(self):
        self.fileList = [self.srcFolder]
        self.loadImages()
        '''
        # load labels
        key = self.sitkImages.keys()[0]
        spacing = self.sitkImages[key][0].GetSpacing()
        manager = LabelManager(self.srcFolder, spacing)
        manager.createLabelFileList()
        self.sitkGT = manager.load_labels(self.label_list)
        '''

    def getNumpyImages(self):
        dat = self.getNumpyData(self.sitkImages,sitk.sitkLinear)
        for key in dat:
            dat[key] = dat[key][0]
        return dat

    def getNumpyGT(self):
        dat = self.getNumpyData(self.sitkGT,sitk.sitkLinear)
        for key in dat:
            dat_list = dat[key]
            num_dat = np.zeros([len(dat_list), self.params['NumVolSize'][0], self.params['NumVolSize'][1],
                            self.params['NumVolSize'][2]], dtype=np.float32)
            for i in range(len(dat_list)):
                num_dat[i,:,:,:] = (dat_list[i]>0.5).astype(dtype=np.float32)
            dat[key] = num_dat
        return dat

    def getNumpyData(self, dat, method):
        ret=dict()
        for key in dat:
            dat_list = dat[key]
            result_list = []
            for i in range(len(dat_list)):
                img = dat_list[i]
                # we rotate the image according to its transformation using the direction and according to the final spacing we want
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
                    T = sitk.AffineTransform(3)
                    T.SetMatrix(img.GetDirection())
                    resampler.SetTransform(T.GetInverse())

                imgResampled = resampler.Execute(img)
                imgCentroid = np.asarray(newSize, dtype=float) / 2.0
                imgStartPx = (imgCentroid - self.params['NumVolSize'] / 2.0).astype(dtype=int)
                regionExtractor = sitk.RegionOfInterestImageFilter()
                regionExtractor.SetSize(list(self.params['NumVolSize'].astype(dtype=int)))
                regionExtractor.SetIndex(list(imgStartPx))

                imgResampledCropped = regionExtractor.Execute(imgResampled)
                result_list.append(np.transpose(sitk.GetArrayFromImage(imgResampledCropped).astype(dtype=float), [1, 2, 0]))
            ret[key] = result_list
        return ret

    def filter(self, dat):
        (w,h,d) = dat.shape
        for i in range(0,d):
            count = np.sum(dat[:,:,i])
            if count < 40:
                dat[:, :, i] = np.zeros((w,h),dtype=float)
        return dat

    def result2Points(self, result, dicomPath):
        result = ndimage.median_filter(result, 9)
        #result = filter(result)
        img = self.sitkImages[dicomPath][0]
        factor = np.asarray([self.params['dstRes'][0], self.params['dstRes'][1], self.params['dstRes'][2]]) \
                 / [img.GetSpacing()[0], img.GetSpacing()[1], img.GetSpacing()[2]]

        newSize = np.asarray(result.shape * factor, dtype=int)
        start = (img.GetSize() - newSize) / 2
        points_list = []
        for i in range(result.shape[2]):
            temp_list = []
            contours = measure.find_contours(np.transpose(result[:, :, i], [1, 0]), 0.1)
            for contour in contours:
                if len(contour) < 20:
                    continue
                points = contour * factor[0:2]
                points += start[0:2]
                points = points * img.GetSpacing()[0:2]
                temp_list.append(points)
            points_list.append((i + start[2], temp_list))
        return points_list

    '''
    def writeResultsFromNumpyLabel(self, result, dicomPath, structureName, sourcePath, destPath):
        result = ndimage.median_filter(result, 9)
        img = self.sitkImages[dicomPath][0]
        factor = np.asarray([self.params['dstRes'][0], self.params['dstRes'][1],self.params['dstRes'][2]]) \
                 / [img.GetSpacing()[0], img.GetSpacing()[1], img.GetSpacing()[2]]

        newSize = np.asarray(result.shape * factor, dtype=int)
        start = (img.GetSize() - newSize) / 2
        point_list = []
        for i in range(result.shape[2]):
            contours = measure.find_contours(np.transpose(result[:,:,i], [1, 0]), 0.3)
            for contour in contours:
                if len(contour) < 20:
                    continue
                points = contour*factor[0:2]
                points += start[0:2]
                points = points*img.GetSpacing()[0:2]
                list.append(points)
            point_list.append((i+start[2], list))
        rtExport = RTExport(dicomPath, sourcePath, destPath)
        rtExport.save(structureName, point_list)
        print "ok"
    '''

