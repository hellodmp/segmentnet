import dicomparser
import numpy as np
from matplotlib.path import Path
from os import listdir
from os.path import isfile, join


def get_contours(structure_file, structure_name):
    rtss = dicomparser.DicomParser(filename=structure_file)
    structures = rtss.GetStructures()
    for i, structure in structures.iteritems():
        if structure['name'] == structure_name:
            return structure['planes']
    return {}

def get_imageData(ct_file):
    ct = dicomparser.DicomParser(filename=ct_file)
    imageData = ct.GetImageData()
    return imageData

#convert old contours to new contours
def contours_convert(contours_dict, imageData):
    newContours = {}
    for key in contours_dict.keys():
        contours = contours_dict[key]
        pixelspacing = imageData["pixelspacing"]
        position = imageData["position"]
        list = []
        for contour in contours:
            list.append([((item[0]-position[0])/pixelspacing[0],(item[1]-position[1])/pixelspacing[1]) for item in contour["contourData"]])
        newContours[key] = list
    return newContours

def get_mask(lines, rows, columns):
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
def contours2mask(contours_dict, imageData):
    mask_dict = {}
    for key in contours_dict.keys():
        contours = contours_dict[key]
        mask = get_mask(contours,imageData["rows"],imageData["columns"])
        mask_dict[key] = mask
    return mask_dict

def get_mask_dict(path):
    structure_file = [f for f in listdir(path) if isfile(join(path, f)) and f.startswith('RS')]
    ct_list = [f for f in listdir(path) if isfile(join(path, f)) and f.startswith('CT')]




if __name__ == '__main__':
    contours = get_contours("./Dataset/V13265/RS.1.2.246.352.71.4.126422491061.161810.20151221162725.dcm","PTV")
    imageData = get_imageData("./Dataset/V13265/CT.1.3.12.2.1107.5.1.4.49611.30000014060506014781200000199.dcm")
    newContours = contours_convert(contours,imageData)
    mask_dict = contours2mask(newContours,imageData)
    print mask_dict