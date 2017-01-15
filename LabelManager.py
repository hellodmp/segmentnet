import numpy as np
import SimpleITK as sitk
from matplotlib.path import Path
from preprocess import dicomparser
from os import listdir
from os.path import isfile, isdir, join

class LabelManager(object):
    srcFolder = None
    spacing = None
    fileList = None

    def __init__(self, srcFolder, spacing):
        self.srcFolder = srcFolder
        self.spacing = spacing

    def get_contours(self, structure_file, structure_name):
        rtss = dicomparser.DicomParser(filename=structure_file)
        structures = rtss.GetStructures()
        for i, structure in structures.iteritems():
            if structure['name'] == structure_name:
                return structure['planes']
        return {}

    def get_imageData(self, ct_file):
        ct = dicomparser.DicomParser(filename=ct_file)
        image_data = ct.GetImageData()
        return image_data

    #convert old contours to new contours
    def contours_convert(self, contours_dict, image_data):
        newContours = {}
        for key in contours_dict.keys():
            contours = contours_dict[key]
            pixelspacing = image_data["pixelspacing"]
            position = image_data["position"]
            list = []
            for contour in contours:
                list.append([((item[0]-position[0])/pixelspacing[0],(item[1]-position[1])/pixelspacing[1]) for item in contour["contourData"]])
            newContours[key] = list
        return newContours

    def get_mask(self, lines, rows, columns):
        mask_list = []
        y_list = range(0,columns)
        for line in lines:
            mask = np.zeros((rows, columns))
            polyline = Path(line)
            for x in range(0,rows):
                points = [(x,y) for y in y_list]
                mask[x,:] = polyline.contains_points(points)
            mask_list.append(mask)
        for i in range(len(mask_list)):
            if i == 0:
                result_mask = mask_list[0]
            else:
                result_mask = np.logical_or(result_mask, mask_list[i])
        return result_mask

    #convert new contours to mask
    def contours2mask(self, contours_dict, image_data):
        mask_dict = {}
        for key in contours_dict.keys():
            contours = contours_dict[key]
            mask = self.get_mask(contours,image_data["rows"],image_data["columns"])
            mask_dict[key] = mask
        return mask_dict

    def get_z_list(self, path):
        ct_list = [path+"/"+f for f in listdir(path) if isfile(join(path, f)) and f.startswith('CT')]
        z_list = []
        for path in ct_list:
            image_data = self.get_imageData(path)
            z_list.append(image_data["position"][2])
        z_list.sort()
        return z_list, image_data

    def get_mask_list(self, path, structure_names, image_data):
        structure_file = [path+"/"+f for f in listdir(path) if isfile(join(path, f)) and f.startswith('RS')]
        mask_list = []
        for structure_name in structure_names:
            contours = self.get_contours(structure_file[0], structure_name)
            if len(contours) > 0:
                new_contours = self.contours_convert(contours, image_data)
                mask = self.contours2mask(new_contours, image_data)
                mask_list.append(mask)
        return mask_list

    def dict2vol(self, imageDict):
        list = sorted(imageDict.iteritems(), key=lambda d: d[0])
        (d, w, h) = list[0][1].shape
        data = np.zeros((len(list), w, h))
        for i in range(len(list)):
            data[i,:,:] = list[i][1]
        image = sitk.GetImageFromArray(data)
        return image

    def createLabelFileList(self):
        self.fileList = [self.srcFolder+"/"+f for f in listdir(self.srcFolder) if isdir(join(self.srcFolder, f))]
        print 'FILE LIST: ' + str(self.fileList)


    def load_labels(self, structure_names=["Bladder"]):
        sitkLabels = {}
        for path in self.fileList:
            z_list, image_data = self.get_z_list(path)
            mask_list = self.get_mask_list(path, structure_names, image_data)
            mask_dict = mask_list[0]
            (w, h) = mask_dict[mask_dict.keys()[0]].shape
            data = np.zeros((len(z_list), w, h))
            for i in range(len(z_list)):
                key = str(z_list[i])+'0'
                if key in mask_dict.keys():
                    data[i,:,:] = mask_dict[key]
                else:
                    data[i,:,:] = np.zeros((w, h))
            sitkLabels[path] = sitk.GetImageFromArray(data)
            sitkLabels[path].SetSpacing(self.spacing)
        return sitkLabels

if __name__ == '__main__':
    manager = LabelManager("./Dataset/data/", np.array([2.0, 2.0, 5.0]))
    manager.createLabelFileList()
    manager.load_labels()


'''
if __name__ == '__main__':
    z_list, mask_list = get_mask_dict("./Dataset/data/V13265/", ["Bladder"])
    print z_list, mask_list
    for mask_dict in mask_list:
        for key in mask_dict.keys():
            mask = mask_dict[key]
            print key, np.mean(mask)
        print "\n"
'''