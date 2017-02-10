import dicom

class RTExport(object):
    sourcePath = None
    destPath = None
    dataSet = None
    structureSetList = None
    roiObservationList = None
    roiContourList = None

    def __init__(self, source, dest):
        self.sourcePath = source
        self.destPath = dest
        self.dataSet = dicom.read_file(source)

    def get_by_structure(self, structure_name):
        structureSetList = self.dataSet.StructureSetROIs
        roiObservationList = self.dataSet.RTROIObservations
        roiContourList = self.dataSet.ROIContours
        for i in range(len(structureSetList)):
            if structureSetList[i].ROIName == structure_name:
                structureSet = structureSetList[i]
                roiObservation = roiObservationList[i]
                roiContour = roiContourList[i]
                return structureSet, roiObservation, roiContour
        return None, None, None

    def setStructureSet(self, structureSet):
        return

    def setRoiObservation(self, roiObservation):
        return

    def setRoiContour(self, roiContour):
        return


if __name__ == "__main__":
    source = "Dataset/data1/V13195/RS.1.2.246.352.71.4.126422491061.189601.20151221094515.dcm"
    dest = "Dataset/rstest.dcm"
    rtExport = RTExport(source, dest)
    #print rtExport.structureSetList
    name = "Urinary Bladder"
    rtExport.get_by_structure(name)
