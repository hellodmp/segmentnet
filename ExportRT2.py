import numpy as np
import itk
import SimpleITK as sitk

def createImage():
    data = np.zeros((100,100),dtype=int)
    label = np.ones((20,20),dtype=int)
    data[20:40,20:40] = label
    data[30:50,50:70] = label
    #data[40:70,40:70,1] = label
    inputImage = sitk.GetImageFromArray(data)
    '''
    Extractor = sitk.ExtractImageFilter()
    Extractor.SetSize( [100,100,0] )
    image = Extractor.SetIndex( [0,0,0] )
    '''
    segmented_surface = sitk.LabelContour(inputImage)
    edges = sitk.GetArrayFromImage(segmented_surface)
    edge_indexes = np.where(edges == 0)
    return edge_indexes

def mask2contour(image):
    ImageType = itk.Image[itk.F, 2]
    ExtractorType = itk.ContourExtractor2DImageFilter[ImageType]
    extractor = ExtractorType.New()
    extractor.SetInput(image)
    extractor.SetContourValue(1)
    extractor.Update()
    print extractor.GetNumberOfOutputs()
    print extractor.GetOutputs()
    [extractor.GetOutput(i) for i in  range(extractor.GetNumberOfOutputs())]
    #itk.PolyLineParametricPath[2].cast(obj)

def test():
    imageType = itk.Image.F3
    buf = np.zeros( (100,100,100), dtype = np.float32)
    itkImage = itk.PyBuffer[imageType].GetImageFromArray(buf)
    return itkImage

if __name__ == '__main__':
    image = createImage()
    #mask = mask2contour(image)