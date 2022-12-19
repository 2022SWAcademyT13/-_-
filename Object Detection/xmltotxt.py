"""
원본 xml을 가져와 yolo의 annotation file로 만들기 위한 작업
"""

# 사용할 라벨 리스트
lblist = ['barricade',
 'bench',
 'bicycle',
 'bollard',
 'bus',
 'car',
 'chair',
 'fire_hydrant',
 'kiosk',
 'motorcycle',
 'movable_signage',
 'person',
 'pole',
 'potted_plant',
 'power_controller',
 'scooter',
 'stop',
 'table',
 'traffic_light',
 'traffic_light_controller',
 'tree_trunk',
 'truck']

# 딕셔너리화
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse, parseString

from tqdm import tqdm
image_dict = dict()
for x in tqdm(xml):
    tree = ET.parse(x)
    root = tree.getroot()
    
    for img in root.findall('image'):
      box = [child.attrib for child in img if child.get('label') in lblist]
      info = img.attrib
      image_dict[img.get('name')[0:-4]] = [box,info]


len(image_dict) # 319712

label_imgename = [name for name in image_dict.keys()] # name만 추출

# txt화
# 딕셔너리에서 필요한 정보 추출 (info)
def get_info(dic, dic2):
    try:
        height = float(dic2['height'])
        width = float(dic2['width'])
        label = lblist.index(dic['label'])
        xtl = float(dic['xtl'])
        ytl = float(dic['ytl'])
        xbr = float(dic['xbr'])
        ybr = float(dic['ybr'])

        xc = .5*(xtl + xbr)/width
        yc = .5*(ytl + ybr)/height
        w = (xbr - xtl)/width
        h = (ybr - ytl)/height
    except:
        label = -1
        xc = 0
        yc = 0
        w = 0
        h = 0

    data = "%d %f %f %f %f\n"%(label, xc, yc, w, h)
    return data

# 추출된 정보에서 txt 만들기
def write_txt(name, txt_path, data):
    f = open(txt_path + "/%s.txt"%name, 'w') # name이름의 txt open
    [f.write(d) for d in data if int(d.split()[0]) > -1]
    f.close()

# 실행
# train, test, validation별로 나눠서 label 저장
from tqdm import tqdm

for k,v in tqdm(image_dict.items()):
    if k in set(train):
        path = 'C:/Users/yoonj/인도보행/Bbox/train/labels'
        data = [get_info(info,v[1]) for info in v[0]]
        write_txt(k, path, data)
    elif k in set(test):
        path = 'C:/Users/yoonj/인도보행/Bbox/test/labels'
        data = [get_info(info,v[1]) for info in v[0]]
        write_txt(k, path, data)
    elif k in set(val):
        path = 'C:/Users/yoonj/인도보행/Bbox/val/labels'
        data = [get_info(info,v[1]) for info in v[0]]
        write_txt(k, path, data)
