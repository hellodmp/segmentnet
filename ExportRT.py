import dicom

def test(path):
    ds = dicom.read_file(path)
    print ds.PatientsName
    contours = ds.ROIContours
    roi_list = ds.StructureSetROIs
    for roi in roi_list:
        print roi
    #ds.PatientsName = "zhang san"
    #ds.save_as("Dataset/data/test.dcm")

def view(path):
    ds = dicom.read_file(path)
    print ds

if __name__ == '__main__':
    path = "Dataset/data/V13142/RS.1.2.246.352.71.4.126422491061.189329.20150624154902.dcm"
    #path = "Dataset/data/test.dcm"
    #path = "Dataset/data/V13142/CT.1.3.12.2.1107.5.1.4.49611.30000014051905545139000000085.dcm"
    test(path)
    #view(path)