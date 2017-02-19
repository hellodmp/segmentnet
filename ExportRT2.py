import numpy as np
import itk
import SimpleITK as sitk
from utilities import *

def createImage():
    data = np.zeros((100,100),dtype=int)
    label = np.ones((20,20),dtype=int)
    data[20:40,20:40] = label
    data[30:50,50:70] = label
    inputImage = sitk.GetImageFromArray(data)
    '''
    Extractor = sitk.ExtractImageFilter()
    Extractor.SetSize( [100,100,0] )
    image = Extractor.SetIndex( [0,0,0] )
    '''
    segmented_surface = sitk.LabelContour(inputImage)
    edges = sitk.GetArrayFromImage(segmented_surface)
    edge_indexes = np.where(edges == 1)
    points = [([int(x), int(y)]) for y, x in zip(edge_indexes[0], edge_indexes[1])]
    return points

def pointSort(points):
    start_point = points[0]
    result_list = [[]]
    threshold = 10
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
    return result_list

def mask2contour(image):
    ImageType = itk.Image[itk.F, 2]
    ExtractorType = itk.ContourExtractor2DImageFilter[ImageType]
    extractor = ExtractorType.New()
    extractor.SetInput(image)
    extractor.SetContourValue(1)
    extractor.Update()
    print extractor.GetNumberOfOutputs()
    print extractor.GetOutputs()
    [extractor.GetOutput(i) for i in range(extractor.GetNumberOfOutputs())]
    #itk.PolyLineParametricPath[2].cast(obj)

def test():
    imageType = itk.Image.F3
    buf = np.zeros( (100,100,100), dtype = np.float32)
    itkImage = itk.PyBuffer[imageType].GetImageFromArray(buf)
    return itkImage

if __name__ == '__main__':
    points = createImage()
    list = pointSort(points)
    print len(list)
    #mask = mask2contour(image)