"""
EDA 결과 image의 label이 불균형하기 때문에 이를 해소하기 위한 작업
"""

import pickle
from tqdm import tqdm

with open('/content/drive/MyDrive/xml_dictionary.dat',"rb") as f:
    xmldict = pickle.load(f)

label_imgename = [name for name in xmldict.keys()]
def choose(key,num):
    i = 0
    b = list()
    for name in label_imgename:
        if i < num:
            for _ in xmldict[name][0]:
                if _['label'] == key:
                    b.append(name)
                    i += 1
                    break
    return b, i
D = dict()
S = set()
for _ in tqdm(range(17)):
    a, b = choose(_,1000)
    D[_] = b
    S  = S | set(a)
len(S)
