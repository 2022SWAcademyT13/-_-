"""
image가 저장되어 있는 폴더가 여러개이기 때문에 하나로 묶기 위한 작업
"""
from glob import glob

folder_path = glob('/content/drive/MyDrive/인도보행 영상/바운딩박스/*') # bounding box path
print(len(folder_path))

folder_path = [path for path in folder_path if path not in zip_path]

from os import listdir
def fileids(path, ext = None):
    fileList = list()
    path = path if path[-1] == '/' else path + '/'
    for fileName in listdir(path):
        if fileName.endswith(ext):
            fileList.append(path + fileName)
    return fileList

jpg_path = [fileids(path, ext = 'jpg') for path in folder_path]
xml_path = [fileids(path, ext = 'xml') for path in folder_path]
png_path = [fileids(path, ext = 'png') for path in folder_path]

img_path = [jpg + png for jpg,png in zip(jpg_path, png_path)]

img_path
