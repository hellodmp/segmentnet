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

def get_mask_dict(path,structure_names):
    structure_file = [path+f for f in listdir(path) if isfile(join(path, f)) and f.startswith('RS')]
    ct_list = [path+f for f in listdir(path) if isfile(join(path, f)) and f.startswith('CT')]
    z_list = []
    for path in ct_list:
        imageData = get_imageData(path)
        z_list.append(imageData["position"][2])
    z_list.sort()

    mask_list = []
    for structure_name in structure_names:
        contours = get_contours(structure_file[0], structure_name)
        if len(contours) > 0:
            new_contours = contours_convert(contours, imageData)
            mask = contours2mask(new_contours, imageData)
            mask_list.append(mask)
    return z_list,mask_list


if __name__ == '__main__':
    z_list, mask_list = get_mask_dict("./Dataset/V13265/", ["PTV","Bladder","FemoralHead"])
    print z_list, mask_list
    for mask_dict in mask_list:
        for key in mask_dict.keys():
            mask = mask_dict[key]
            print key, np.mean(mask)
        print "\n"
