import dicom
import copy

class RTExport(object):
    sourcePath = None
    destPath = None
    sourceDs = None
    destDs = None
    structureSetList = None
    roiObservationList = None
    roiContourList = None

    def __init__(self, source, dest):
        self.sourcePath = source
        self.destPath = dest
        self.sourceDs = dicom.read_file(source)
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

    def modifyStructureSet(self, structureSet):
        structureSet.ROIName = "AUTO " + structureSet.ROIName
        return structureSet

    def modifyRoiObservation(self, roiObservation):
        return roiObservation

    def modifyRoiContour(self, roiContour):
        return roiContour

    def addNewLabel(self, name):
        structureSet, roiObservation, roiContour = self.get_by_structure(name)
        structureSet = self.modifyStructureSet(structureSet)
        structureSetList = self.destDs.StructureSetROIs
        structureSetList.append(structureSet)

        roiObservation = self.modifyRoiObservation(roiObservation)
        roiObservationList = self.destDs.RTROIObservations
        roiObservationList.append(roiObservation)

        roiContour = self.modifyRoiContour(roiContour)
        roiContourList = self.destDs.ROIContours
        roiContourList.append(roiContour)
        print "ok"

    def save(self):
        self.destDs.save_as(self.destPath)

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
    source = "Dataset/data/V13195/RS.1.2.246.352.71.4.126422491061.189601.20151221094515.dcm"
    dest = "Dataset/data/test.dcm"
    rtExport = RTExport(source, dest)
    #print rtExport.structureSetList
    name = "Urinary Bladder"
    rtExport.addNewLabel(name)
    rtExport.save()
