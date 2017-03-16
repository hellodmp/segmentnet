#coding=utf-8

import shutil
import dicomparser
import os

def part(path):
    seriesDict = {}
    sop_dict={"ct","rtss","rtdose","rtplan"}
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        dcm = dicomparser.DicomParser(filename=path)
        sopclass = dcm.GetSOPClassUID()
        if sopclass not in sop_dict:
            continue
        seriesId = dcm.GetSeriesInfo()['study']
        if seriesId in seriesDict:
            seriesDict[seriesId].append(path)
        else:
            seriesDict[seriesId] = [path]
    return seriesDict

def containRtss(path_list):
    result = False
    structures_dict = {"Eye_L","Eye_R","Optic Nerve_L","Optic Nerve_R","Parotid_L","Parotid_R","Brainstem","mouth","Neck"}
    for path in path_list:
        dcm = dicomparser.DicomParser(filename=path)
        sopclass = dcm.GetSOPClassUID()
        if sopclass != "rtss":
            continue
        structures = dcm.GetStructures()
        count = 0
        for key in structures.keys():
            if structures[key]["name"] in structures_dict:
                count += 1
        if count > 5:
            result = True
        else:
            path_list.remove(path)
    return result, path_list


def copy(key, path_list, destPath):
    strs = key.split(".")
    series_path = destPath + "/" + strs[-2]
    os.mkdir(series_path)
    print "create " + series_path
    for source in path_list:
        shutil.copy(source, series_path)


def getSeries(path):
    dcm = dicomparser.DicomParser(filename=path)
    sopclass = dcm.GetSOPClassUID()
    seriesId = dcm.GetSeriesInfo()['study']
    if sopclass == "rtss":
        print path, seriesId

def getstructure(path):
    dcm = dicomparser.DicomParser(filename=path)
    structures = dcm.GetStructures()
    for key in structures.keys():
        print structures[key]["name"]


if __name__ == "__main__":
    source = "/home/andrew/Desktop/source"
    destPath = "/home/andrew/Desktop/dest"
    #print os.listdir(source)
    for filename in os.listdir(source):
        print filename
        dir = os.path.join(source, filename)
        if not os.path.isdir(dir):
            continue
        seriesDict = part(dir)
        for key in seriesDict.keys():
            path_list = seriesDict[key]
            result, path_list = containRtss(path_list)
            if result:
                copy(key, path_list, destPath)
