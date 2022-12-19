"""
원본 image path를 train, test, validation으로 나눈 후 
하나의 폴더에 모으기 위한 코드
"""
import pandas as pd

train = pd.read_csv("/content/drive/MyDrive/train_name.csv")
test = pd.read_csv("/content/drive/MyDrive/test_name.csv")
val = pd.read_csv("/content/drive/MyDrive/val_name.csv")

train.columns = ['index', 'name']
test.columns = ['index', 'name']
val.columns = ['index', 'name']

train = list(train['name'])
test = list(test['name'])
val = list(val['name'])

import shutil
from tqdm import tqdm


from_path = "/content/drive/MyDrive/images_31만개짜리"

to_path = "/content/drive/MyDrive/Data/train_/image"
[shutil.copy(from_path + '/' + img + '.jpg', to_path + '/' + img + '.jpg') for img in tqdm(train)]

import shutil
from tqdm import tqdm

from_path = "/content/drive/MyDrive/images_31만개짜리"

to_path = "/content/drive/MyDrive/Data/test/image"
[shutil.copy(from_path + '/' + img + '.jpg', to_path + '/' + img + '.jpg') for img in tqdm(test)]

import shutil
from tqdm import tqdm

from_path = "/content/drive/MyDrive/images_31만개짜리"

to_path = "/content/drive/MyDrive/Data/validation/image"
[shutil.copy(from_path + '/' + img + '.jpg', to_path + '/' + img + '.jpg') for img in tqdm(validation)]
