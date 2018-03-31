from PIL import Image
from os import listdir
import numpy as np

path = '/home/hscuabc/WorkSpace/Python-srf/吴恩达课程代码实现/data/negdata'
save_path = '/home/hscuabc/WorkSpace/Python-srf/吴恩达课程代码实现/data/negdatatojpg'
w_expect = 20;l_expect = 20


def reshape_img(Img_path, save_img_path):
    im = Image.open(Img_path)
    expect_img = im.resize((w_expect, l_expect), Image.ANTIALIAS)
    expect_img.save(save_img_path)
    return


def all_file_reshape(file_path, save_path):
    all_file = listdir(file_path)
    cnt = 0
    for file in all_file:
        Img_path = file_path + '/' + file
        save_img_path = save_path + '/' + file[:-4] + '.jpg'
        reshape_img(Img_path, save_img_path)
        cnt = cnt + 1 
        print(cnt)

all_file_reshape(path, save_path)