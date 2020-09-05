import CNN_basic as cnn 
import tensorflow as tf 
import numpy as np
import pickle as pk
import os

PATH = 'D:/TWINKLE_STAR_DOWNLOAD/cifar-10-python/cifar-10-batches-py'

def get_dataset(file):
    with open(file, 'rb') as fo:
        data = pk.load(fo, encoding='bytes')
    labels = data[b'labels']
    imgs = data[b'data']    
    return labels,imgs

def main():
    dir = os.listdir(PATH)
    x = []
    y = []
    for file in dir:
        labels,imgs = get_dataset(PATH+'/'+file)
        imgs = imgs.tolist()
        x.append(imgs)
        y.append(labels)
        
        test_x = []
        test_y = []
        
    for i in range(5):
        for j in range(10000):
            x[i][j] = np.reshape(x[i][j], (3,32,32))
            x[i][j] = x[i][j].transpose((1,2,0))
            x1 = tf.image.random_flip_left_right(x[i][j])
            x2 = tf.image.random_brightness(x1,max_delta=63)
            x3 = tf.image.random_contrast(x2,lower=0.2,upper=1.8)
            x4 = tf.image.per_image_standardization(x3)
            with tf.Session():
                x4 = x4.eval()
            test_x.append(x4)
            test_y.append(y[i][j])
            if j%10 == 0: print('%d'%(j))
        x.append(test_x)
        y.append(test_y)
        test_x = []
        test_y = []
    np.save('newbatch_x', x)
    np.save('newbatch_y', y)
    
if __name__ == "__main__":
    main()