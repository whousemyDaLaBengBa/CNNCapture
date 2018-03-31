from PIL import Image
from os import listdir
import numpy as np

POSPATH = '/home/hscuabc/WorkSpace/Python-srf/吴恩达课程代码实现/data/posdata'
NEGPATH = '/home/hscuabc/WorkSpace/Python-srf/吴恩达课程代码实现/data/negdata'

CNTP = 0
CNTN = 0

def trans_img_str(img):
    r, g, b = img.split()
    x,y = img.size
    len = x * y
    r_arr = np.array(r).reshape(len)
    g_arr = np.array(g).reshape(len)
    b_arr = np.array(b).reshape(len)
    img_arr = 0.2989 * r_arr + 0.5870 * g_arr + 0.1140 * b_arr
    print(len)
    return img_arr.tolist()

def get_img(num):
    global CNTP
    global CNTN

    if num % 2 == 1:
        CNTP = CNTP + 1
        img_path = POSPATH + '/' + str(CNTP) + '.jpg'
        img = Image.open(img_path)
        Y = [1.0]
    else:
        CNTN = CNTN + 1
        img_path = NEGPATH + '/' + str(CNTN) + '.jpg'
        img = Image.open(img_path)
        Y = [0.0]

    X = trans_img_str(img)
    return X,Y


def get_next_train_branch(img_num = 256):
    listX = [None] * img_num
    listY = [None] * img_num
    for k in range(img_num):
        listX[k],listY[k]=get_img(k)

    return listX,listY

'''
listX,listY = get_next_train_branch(3)
print(listX)
print(listY)
'''
    


