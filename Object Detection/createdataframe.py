from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

def attrlistmake2(Xml):
    data = open(
                    Xml,
                    'r',
                    encoding='utf-8'
                ).read()
    dom = BeautifulSoup(data, features='xml')
    
    attrlist2 = [[a.attrs,[b.attrs for b in a.findChildren()]] for a in dom.select('image')]
            
    return attrlist2
# 더 빠른 방법
def tmake2(attrlist2):
    T = [[dict(_[0],**__) for __ in _[1]] for _ in attrlist2]
    T = np.concatenate(T).tolist()
    T = pd.DataFrame(T)
    return T

import os

dir_path = ".\인도보행\인도보행 영상\바운딩박스\Bbox_1_new\Box_1" # xml들이 들어있는 폴더 경로(\ 대신 / 사용)
xml_list = []

for (root, directories, files) in os.walk(dir_path):
#     for d in directories:
#         d_path = os.path.join(root, d)
#         print(d_path)

    for file in files:
        file_path = os.path.join(root, file)
        xml_list.append(file_path) # xml_list에 각 xml 파일이이 원소로 들어감
#         print(file_path)

T = None
i = 0
for _ in xml_list:
    if T is None:
        T = tmake2(attrlistmake2(_))
    else:
        T = pd.concat([T,tmake2(attrlistmake2(_))])
    print(i)
    i+=1

T.reset_index()
T.to_csv("T.csv")
T.to_pickle("T.pkl")
