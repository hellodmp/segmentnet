import dicom
import copy
from os import listdir
from os.path import isfile, isdir, join, splitext
from scipy import ndimage
from skimage import measure
import numpy as np

class RTExport(object):
    dicomPath = None
    sourceStructurePath = None
    destStructurePath = None
    sourceDs = None
    destDs = None
    structureSetList = None
    roiObservationList = None
    roiContourList = None

    def __init__(self, dicomPath, sourceStructurePath, destStructurePath):
        self.dicomPath = dicomPath
        self.destStructurePath = destStructurePath
        self.sourceStructurePath = sourceStructurePath
        #self.structurePath = [dicomPath + "/" + f for f in listdir(dicomPath) if isfile(join(dicomPath, f)) and f.startswith('RS')]
        self.sourceDs = dicom.read_file(self.sourceStructurePath)
        self.destDs = copy.deepcopy(self.sourceDs)

    def get_by_structure(self, structure_name):
        structureSetList = self.sourceDs.StructureSetROIs
        roiObservationList = self.sourceDs.RTROIObservations
        roiContourList = self.sourceDs.ROIContours
        for i in range(len(structureSetList)):
            if structureSetList[i].ROIName == structure_name:
                structureSet = copy.deepcopy(structureSetList[i])
                roiObservation = copy.deepcopy(roiObservationList[i])
                roiContour = copy.deepcopy(roiContourList[i])
                return structureSet, roiObservation, roiContour
        return None, None, None

    def modifyStructureSet(self, structureSet,roiNumber):
        structureSet.ROIName += " AUTO"
        structureSet.ROINumber = roiNumber
        return structureSet

    def modifyRoiObservation(self, roiObservation, roiNumber):
        roiObservation.ObservationNumber = roiNumber
        roiObservation.RefdROINumber = roiNumber
        roiObservation.ReferencedROINumber = roiNumber
        return roiObservation

    #label_list=[(sopInstanceUID,[(x,y,z),(x,y,z)]),(sopInstanceUID,[(x,y,z),(x,y,z)])]
    def modifyRoiContour(self, roiContour, roiNumber, label_list):
        roiContour.RefdROINumber = roiNumber
        roiContour.ReferencedROINumber = roiNumber
        contour = roiContour.Contours[0]
        contour_list = []
        for label in label_list:
            (sopInstanceUID, points_list) = label
            contour_temp = copy.deepcopy(contour)
            contour_temp.ContourImages[0].ReferenedSOPInstanceUID = sopInstanceUID
            contour_temp.NumberofContourPoints = len(points_list)
            contour_temp.ContourData = []
            contour_data = []
            for item in points_list:
                contour_data.append(item[0])
                contour_data.append(item[1])
                contour_data.append(item[2])
            contour_temp.ContourData = contour_data
            contour_list.append(contour_temp)
        roiContour.Contours = contour_list
        return roiContour

    def addNewLabel(self, name, label_list):
        roiNumber = 0
        for structureSet in self.destDs.StructureSetROIs:
            if structureSet.ROINumber > roiNumber:
                roiNumber = structureSet.ROINumber
        roiNumber += 1
        structureSet, roiObservation, roiContour = self.get_by_structure(name)

        structureSet = self.modifyStructureSet(structureSet,roiNumber)
        structureSetList = self.destDs.StructureSetROIs
        structureSetList.append(structureSet)

        roiObservation = self.modifyRoiObservation(roiObservation,roiNumber)
        roiObservationList = self.destDs.RTROIObservations
        roiObservationList.append(roiObservation)

        roiContour = self.modifyRoiContour(roiContour,roiNumber,label_list)
        roiContourList = self.destDs.ROIContours
        roiContourList.append(roiContour)
        #print roiContour

    #image_list=[(sliceLocation,sopInstanceUID,startPosition),(sliceLocation,sopInstanceUID,startPosition)]
    def getImageList(self):
        path = self.dicomPath
        path_list = [path+"/"+f for f in listdir(path) if isfile(join(path, f)) and f.startswith('CT')]
        image_list = []
        for path in path_list:
            image_data = dicom.read_file(path)
            sliceLocation = image_data.SliceLocation
            sopInstanceUID = image_data.SOPInstanceUID
            startPosition = image_data.ImagePositionPatient
            image_list.append((sliceLocation,sopInstanceUID,startPosition))
        image_list.sort()
        sorted(image_list, key=lambda data : data[0])
        return image_list

    '''
    def createLabelList(self, image_list, points_list):
        label_list = []
        for (index,points) in points_list:
            (sliceLocation, sopInstanceUID, startPosition) = image_list[index]
            points = [(point[0]+startPosition[0],point[1]+startPosition[1],startPosition[2]) for point in points]
            label_list.append((sopInstanceUID,points))
        return label_list
    '''


    def createLabelList(self, image_list, points_list):
        label_list = []
        for (index,list) in points_list:
            for points in list:
                (sliceLocation, sopInstanceUID, startPosition) = image_list[index]
                points = [(point[0]+startPosition[0],point[1]+startPosition[1],startPosition[2]) for point in points]
                label_list.append((sopInstanceUID,points))
        return label_list

    def addStructure(self,structureName, points_list):
        image_list = self.getImageList()
        label_list = self.createLabelList(image_list, points_list)
        self.addNewLabel(structureName, label_list)

    def save(self):
        self.destDs.save_as(self.destStructurePath)

'''
if __name__ == "__main__":
    source = "Dataset/data1/V13195/RS.1.2.246.352.71.4.126422491061.189601.20151221094515.dcm"
    dest = "Dataset/rstest.dcm"
    rtExport = RTExport(source, dest)
    #print rtExport.structureSetList
    name = "Urinary Bladder"
    rtExport.get_by_structure(name)
'''

if __name__ == "__main__":
    source = "Dataset/data1/RS.1.2.246.352.71.4.126422491061.189601.20151221094515.dcm"
    dest = "Dataset/data1/test.dcm"
    rtExport = RTExport(source, dest)
    #print rtExport.structureSetList
    name = "Urinary Bladder"
    label_list = [("1.3.12.2.1107.5.1.4.49611.30000014052605590490600002095",[(-45,-234,-381.0),(20,-234,-381.0),(20,-200,-381.0),(-45,-200,-381.0)]),
                  ("1.3.12.2.1107.5.1.4.49611.30000014052605590490600002096",[(-45,-234,-386.0),(20,-234,-386.0),(20,-200,-386.0),(-45,-200,-386.0)])]
    rtExport.addNewLabel(name,label_list)
    rtExport.save()
