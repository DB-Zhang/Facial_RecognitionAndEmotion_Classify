import numpy as np
import cv2
import pandas as pd
import random
import os
 
curdir = os.path.abspath(os.path.dirname(__file__))
 
def gen_record(csvfile,channel):
    data = pd.read_csv(csvfile,delimiter=',',dtype='a')
    labels = np.array(data['emotion'],np.float)
 
    imagebuffer = np.array(data['pixels'])
    #删掉空格，每个图片转化为数组
    images = np.array([np.fromstring(image,np.uint8,sep=' ') for image in imagebuffer])
    #s释放临时buff
    del imagebuffer
 
    #最后一个维度的大小
    num_shape = int(np.sqrt(images.shape[-1]))
 
    #调整数组为48*48图片
    images.shape = (images.shape[0],num_shape,num_shape)
 
    # 三种Training，PublicTest，PrivateTest
    dirs = set(data['Usage'])
 
    class_dir = {}
 
 
    for dr in dirs:
        dest = os.path.join(curdir,dr)
        class_dir[dr] = dest
        if not os.path.exists(dest):
            os.mkdir(dest)
 
    data = zip(labels,images,data['Usage'])
 
    for d in data:
        label = int(d[0])
        #根据标签存放图片到不同文件夹
        destdir = os.path.join(class_dir[d[2]],str(label))
 
        if not os.path.exists(destdir):
            os.mkdir(destdir)
 
        img = d[1]
        filepath = unique_name(destdir,d[2])
        print('Write image to %s' % filepath)
 
        if not filepath:
            continue
 
        sig = cv2.imwrite(filepath,img)
        if not sig:
            print('Error')
            exit(-1)
 
 
def unique_name(pardir,prefix,suffix='jpg'):
    #生成随机文件名
    filename = '{0}_{1}.{2}'.format(prefix,random.randint(1,10**8),suffix)
    filepath = os.path.join(pardir,filename)
    if not os.path.exists(filepath):
        return filepath
    unique_name(pardir,prefix,suffix)
 
 
if __name__ == '__main__':
    filename = './fer2013.csv'
    filename = os.path.join(curdir,filename)
    gen_record(filename,1)
