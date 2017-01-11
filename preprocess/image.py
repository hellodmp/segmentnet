from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import io
from skimage import data


def loadImages(fileList):
    sitkImages = dict()
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(1)
    rescalFilt.SetOutputMinimum(0)

    stats = sitk.StatisticsImageFilter()
    m = 0.
    for f in fileList:
        sitkImages[f] = rescalFilt.Execute(sitk.Cast(sitk.ReadImage(f), sitk.sitkFloat32))
        #sitk_show(sitkImages[f])
        show(sitkImages[f])
        stats.Execute(sitkImages[f])
        m += stats.GetMean()

    meanIntensityTrain = m / len(sitkImages)

def read_image(path):
    #read dicom
    data = sitk.ReadImage(path)

    #Rescale the vlaue into [0,255]
    #filter = SimpleITK.RescaleIntensityImageFilter()
    #dcm = filter.Execute(data,0,255)
    # tansform to numpy
    nda = sitk.GetArrayFromImage(data)
    (w, h, d)=nda.shape
    image = nda.reshape((w, h, d)).transpose(1, 2, 0)
    #image = image[:,:,0]
    return image

def show(image):
    io.imshow(image)
    io.show()

def sitk_show(nda, title=None, margin=0.0, dpi=40):
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi

    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    for k in range(0, nda.shape[2]):
        print "printing slice " + str(k)
        ax.imshow(np.squeeze(nda[:, :, k]), extent=extent, interpolation=None)
        plt.draw()
        plt.pause(0.1)
        plt.waitforbuttonpress()

if __name__ == "__main__":
    path = "../Dataset/V13265/"
    ct_list = [path + f for f in listdir(path) if isfile(join(path, f)) and f.startswith('CT')]
    for i in range(len(ct_list)):
        image = read_image(ct_list[i])
        #show(image)
        sitk_show(image)