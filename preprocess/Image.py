from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import io
from skimage import data
from preprocess import dicomparser

#http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/03_Image_Details.html


def get_imageData(ct_file):
    ct = dicomparser.DicomParser(filename=ct_file)
    print ct.GetSeriesInfo()
    imageData = ct.GetImageData()
    return imageData

def read_images(fileList):
    image_dict = dict()
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(1)
    rescalFilt.SetOutputMinimum(0)
    for path in fileList:
        info = get_imageData(path)
        image = rescalFilt.Execute(sitk.Cast(sitk.ReadImage(path), sitk.sitkFloat32))
        image_dict[info["position"][2]] = sitk.GetArrayFromImage(image)
    return image_dict

def read_series(dir):
    reader = sitk.ImageSeriesReader()
    series_list =  reader.GetGDCMSeriesIDs(dir)
    for series_id in series_list:
        dicom_names = reader.GetGDCMSeriesFileNames(dir, series_id)
        if len(dicom_names) > 1:
            break
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def read_image(path):
    #read dicom
    data = sitk.ReadImage(path, sitk.sitkFloat32)
    filter = sitk.RescaleIntensityImageFilter()
    data = filter.Execute(data,0,1)
    nda = sitk.GetArrayFromImage(data)
    (d, w, h)=nda.shape
    image = nda.reshape((d,w,h)).transpose(1, 2, 0)
    return image


def dict2vol(imageDict):
    list = sorted(imageDict.iteritems(), key=lambda d: d[0])
    (d, w, h) = list[0][1].shape
    data = np.zeros((len(list), w, h))
    for i in range(len(list)):
        data[i,:,:] = list[i][1]
    image = sitk.GetImageFromArray(data)
    return image

def convertNumpyData(img, volSize, dstRes, method=sitk.sitkLinear):
    # we rotate the image according to its transformation using the direction and according to the final spacing we want
    factor = np.asarray(img.GetSpacing()) / [dstRes[0], dstRes[1], dstRes[2]]
    factorSize = np.asarray(img.GetSize() * factor, dtype=float)
    newSize = np.max([factorSize, volSize], axis=0)
    newSize = newSize.astype(dtype=int)

    T = sitk.AffineTransform(3)
    T.SetMatrix(img.GetDirection())
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing([dstRes[0], dstRes[1], dstRes[2]])
    resampler.SetSize(newSize)
    resampler.SetInterpolator(method)
    '''
    if params['normDir']:
        resampler.SetTransform(T.GetInverse())
    '''
    imgResampled = resampler.Execute(img)
    imgCentroid = np.asarray(newSize, dtype=float) / 2.0
    imgStartPx = (imgCentroid - volSize / 2.0).astype(dtype=int)
    regionExtractor = sitk.RegionOfInterestImageFilter()
    regionExtractor.SetSize(list(volSize.astype(dtype=int)))
    regionExtractor.SetIndex(list(imgStartPx))
    imgResampledCropped = regionExtractor.Execute(imgResampled)

    ret = np.transpose(sitk.GetArrayFromImage(imgResampledCropped).astype(dtype=float), [2, 1, 0])
    return ret


def sitk_show(nda, title=None, margin=0.0, dpi=40):
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi

    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
    for k in range(0, nda.shape[2]):
        print "printing slice " + str(k)
        ax.imshow(np.squeeze(nda[:, :, k]),cmap ='gray', extent=extent, interpolation=None)
        plt.draw()
        #plt.pause(1)
        plt.waitforbuttonpress()

if __name__ == "__main__":
    path = "../Dataset/V13265"
    ct_list = [path +"/"+ f for f in listdir(path) if isfile(join(path, f)) and f.startswith('CT')]
    '''
    for file in ct_list:
        get_imageData(file)
    '''
    read_series(path)
    '''
    ct_list = [path + f for f in listdir(path) if isfile(join(path, f)) and f.startswith('CT')]
    images = read_images(ct_list)
    image = dict2vol(images)
    volSize = np.asarray([128,128,64],dtype=int)
    dstRes = np.asarray([1,1,5],dtype=float)
    data = convertNumpyData(image,volSize,dstRes)
    print data.shape
    '''
