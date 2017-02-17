import numpy as np
import SimpleITK as sitk
from LabelManager import LabelManager
from os import listdir
from os.path import isfile, isdir, join, splitext
import utilities
from RTExport import RTExport

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
        self.sitkGT = manager.load_labels(self.label_list)

    def loadTestData(self):
        self.createImageFileList()
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


    def pointSort(self, points):
        start_point = points[0]
        result_list = [[]]
        threshold = 40
        count = len(points)
        result_index = 0
        for i in range(count):
            index = 0
            min_value = 10000
            for j in range(len(points)):
                dis = abs(start_point[0] - points[j][0]) + abs(start_point[1] - points[j][1])
                if dis < min_value:
                    min_value = dis
                    index = j
            if min_value > threshold:
                result_list.append([points[0]])
                result_index = len(result_list) -1
                start_point = points[0]
                del points[0]
            else :
                result_list[result_index].append(points[index])
                start_point = points[index]
                del points[index]
        #result_list = left_list.extend(right_list)
        return result_list[0]

    def transfer_contour(self, edges, edge_indexes):
        index_list = []
        index = edge_indexes[0][0]
        start = 0
        for i in range(len(edge_indexes[0])):
            if edge_indexes[0][i] != index:
                index_list.append((index, start, i - start))
                start = i
                index = edge_indexes[0][start]
        index_list.append((index, start, len(edge_indexes[0]) - start))

        points_list = []
        for (index, start, count) in index_list:
            sub_edges0 = edge_indexes[0][start:start + count]
            sub_edges1 = edge_indexes[1][start:start + count]
            sub_edges2 = edge_indexes[2][start:start + count]
            # Note the reversed order of access between SimpleITK and numpy (z,y,x)
            points = [edges.TransformIndexToPhysicalPoint([int(x), int(y), int(z)]) \
                               for z, y, x in zip(sub_edges0, sub_edges1, sub_edges2)]
            points = self.pointSort(points)
            points_list.append((index,points))
        return points_list


    def writeResultsFromNumpyLabel(self, result, key):
        img = self.sitkImages[key][0]

        factor = np.asarray([self.params['dstRes'][0], self.params['dstRes'][1],
                             self.params['dstRes'][2]]) / [img.GetSpacing()[0], img.GetSpacing()[1], img.GetSpacing()[2]]
        factorSize = np.asarray(result.shape * factor, dtype=float)
        newSize = np.max([factorSize, self.params['NumVolSize']], axis=0)
        newSize = newSize.astype(dtype=int)
        label = np.zeros(img.GetSize(), dtype=np.float32)

        result_image = sitk.GetImageFromArray(np.transpose(result, [2, 1, 0]))
        result_image.SetSpacing(self.params['dstRes'])

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(result_image)
        resampler.SetOutputSpacing(img.GetSpacing())
        resampler.SetSize(newSize)
        resampler.SetInterpolator(sitk.sitkLinear)
        imgResampled = resampler.Execute(result_image)

        imgNum = np.transpose(sitk.GetArrayFromImage(imgResampled).astype(dtype=float), [1, 2, 0])
        start = (img.GetSize() - np.array(imgNum.shape)) / 2
        end = start + imgNum.shape
        label[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = imgNum
        result_image = sitk.GetImageFromArray(np.transpose(label, [2, 1, 0]))
        result_image.SetSpacing(img.GetSpacing())

        #utilities.sitk_show(label)
        thfilter = sitk.BinaryThresholdImageFilter()
        thfilter.SetInsideValue(1)
        thfilter.SetOutsideValue(0)
        thfilter.SetLowerThreshold(0.5)
        toWrite = thfilter.Execute(result_image)

        # connected component analysis
        cc = sitk.ConnectedComponentImageFilter()
        toWritecc = cc.Execute(sitk.Cast(toWrite, sitk.sitkUInt8))
        #result_image = np.transpose(sitk.GetArrayFromImage(toWritecc).astype(dtype=float), [1, 2, 0])
        #utilities.sitk_show(result_image)

        edges = sitk.CannyEdgeDetection(sitk.Cast(toWritecc, sitk.sitkFloat32),
                                        lowerThreshold=0.5, upperThreshold=1.0)
        edge_indexes = np.where(sitk.GetArrayFromImage(edges) == 1.0)
        point_list = self.transfer_contour(edges, edge_indexes)
        source = "Dataset/Test/V16609"
        dest = "Dataset/Test/test.dcm"
        key = "Urinary Bladder"
        rtExport = RTExport(source,dest)
        rtExport.save(key,point_list)
        print "ok"


    '''
    def writeResultsFromNumpyLabel(self, result, key):
        img = self.sitkImages[key][0]

        factor = np.asarray([self.params['dstRes'][0], self.params['dstRes'][1],
                             self.params['dstRes'][2]]) / [img.GetSpacing()[0], img.GetSpacing()[1], img.GetSpacing()[2]]
        factorSize = np.asarray(result.shape * factor, dtype=float)
        newSize = np.max([factorSize, self.params['NumVolSize']], axis=0)
        newSize = newSize.astype(dtype=int)
        label = np.zeros(img.GetSize(), dtype=np.float32)
        start = (img.GetSize() - newSize) / 2
        end = start + result.shape
        label[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = result

        result_image = sitk.GetImageFromArray(np.transpose(label, [2, 1, 0]))
        result_image.SetSpacing(self.params['dstRes'])

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(result_image)
        resampler.SetOutputSpacing(img.GetSpacing())
        resampler.SetSize(newSize)
        resampler.SetInterpolator(sitk.sitkLinear)
        #utilities.sitk_show(num_data)
        thfilter = sitk.BinaryThresholdImageFilter()
        thfilter.SetInsideValue(1)
        thfilter.SetOutsideValue(0)
        thfilter.SetLowerThreshold(0.5)
        toWrite = thfilter.Execute(result_image)
        # connected component analysis
        cc = sitk.ConnectedComponentImageFilter()
        toWritecc = cc.Execute(sitk.Cast(toWrite, sitk.sitkUInt8))

        #arrCC = np.transpose(sitk.GetArrayFromImage(toWritecc).astype(dtype=float), [2, 1, 0])
        #utilities.sitk_show(arrCC)
        #edges = sitk.CannyEdgeDetection(sitk.Cast(toWritecc, sitk.sitkFloat32), lowerThreshold=0.5,upperThreshold=1.0, variance=(1.0, 1.0, 1.0))

        edges = sitk.CannyEdgeDetection(sitk.Cast(toWritecc, sitk.sitkFloat32),
                                        lowerThreshold=0.5,upperThreshold=1.0)
        edge_indexes = np.where(sitk.GetArrayFromImage(edges) == 1.0)

        # Note the reversed order of access between SimpleITK and numpy (z,y,x)
        physical_points = [edges.TransformIndexToPhysicalPoint([int(x), int(y), int(z)]) \
                           for z, y, x in zip(edge_indexes[0], edge_indexes[1], edge_indexes[2])]

        # Setup and solve linear equation system.
        A = np.ones((len(physical_points), 4))
        b = np.zeros(len(physical_points))

        for row, point in enumerate(physical_points):
            A[row, 0:3] = -2 * np.array(point)
            b[row] = -linalg.norm(point) ** 2

        res, _, _, _ = linalg.lstsq(A, b)

        print("The sphere's radius is: {0:.2f}mm".format(np.sqrt(linalg.norm(res[0:3]) ** 2 - res[3])))
        '''



