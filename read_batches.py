import pickle as pk
import numpy as np
import cv2

file = r'D:\TWINKLE_STAR_DOWNLOAD\cifar-10-python\cifar-10-batches-py\data_batch_5'
#依次更改data_batch1~5,导出全部数据
savefile = r'C:\Users\User-SY\Desktop\cifar10\data_batch_5\\'
#自定义存储路径

with open(file, 'rb') as fo:
    data = pk.load(fo, encoding='bytes')

for i in range(10000):
    img = data[b'data'][i]
    img = np.reshape(img, (3,32,32))
    img =img.transpose((1,2,0))
    img_name = str(data[b'filenames'][i])
    img_label = str(data[b'labels'][i])
    cv2.imwrite(savefile+img_label+"_"+img_name+'.jpg',img)
    print("%d pictures successfully out"%(i))