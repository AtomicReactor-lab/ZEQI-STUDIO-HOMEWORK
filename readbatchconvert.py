import pickle as pk
import numpy as np
import cv2
import tensorflow as tf
import os

file = r'D:\TWINKLE_STAR_DOWNLOAD\cifar-10-python\cifar-10-batches-py\data_batch_5'

with open(file, 'rb') as fo:
    data = pk.load(fo, encoding='bytes')

for i in range(10000):
    boy = tf.image.random_flip_up_down(data[b'data'][i])
    boy = tf.image.random_hue(data[b'data'][i])
    boy = tf.image.random_saturation([b'data'][i])
    boy = tf.image.random_brightness([b'data'][i])
    boy = tf.image.random_contrast([b'data'][i])
    if i%100 == 0:
        print('%d00 pictures have been done'%(i%100))

with tf.Session() as sess:
    boy = boy.eval()
    
