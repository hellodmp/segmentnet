import shutil

import dicomparser
import os

def part(path):
    seriesDict = {}
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        dcm = dicomparser.DicomParser(filename=path)
        sopclass = dcm.GetSOPClassUID()
        seriesId = dcm.GetSeriesInfo()['study']
        if seriesId in seriesDict:
            seriesDict[seriesId].append(path)
        else:
            seriesDict[seriesId] = [path]
    return seriesDict

def move(dict, destPath):
    for key in dict.keys():
        strs = key.split(".")
        series_path = destPath + "/" + strs[-2]
        os.mkdir(series_path)
        for source in dict[key]:
            shutil.move(source, series_path)
            print source

def filterDicom(path):
    dcm = dicomparser.DicomParser(filename=path)
    sopclass = dcm.GetSOPClassUID()
    print dcm.GetSeriesInfo()
    if sopclass == "ct" or sopclass == "rtss" or sopclass == "rtdose" or sopclass == "rtplan":
        return True
    else:
        return False

def getSeries(path):
    dcm = dicomparser.DicomParser(filename=path)
    sopclass = dcm.GetSOPClassUID()
    seriesId = dcm.GetSeriesInfo()['study']
    if sopclass == "rtss":
        print path, seriesId


if __name__ == "__main__":
    dir = "/home/andrew/Desktop/chen hou de"
    seriesDict = part(dir)
    move(seriesDict, dir)


'''
if __name__ == "__main__":
    dir = "/home/andrew/Desktop/chen hou de"
    #dirpath, dirnames, filenames = os.walk(dir)
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        if os.path.isfile(path) and filterDicom(path)==False:
            os.remove(path)
'''