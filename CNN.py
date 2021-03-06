import CNN_basic as cnn 
import tensorflow as tf 
import numpy as np
import pickle as pk
import os

PATH = 'D:/TWINKLE_STAR_DOWNLOAD/cifar-10-python/cifar-10-batches-py' #路径下其他文件都要删掉，只能留data_batch1~5

IMAGE_SIZE = 32     #图片大小
NUM_CHANNELS = 3    #图片色通道数

INPUT_NODE = 3072   #32*32*3的图片
FC_SIZE = 512
OUTPUT_NODE = 10     #输出，CIFAR10有10种分类（飞机，汽车，鸟，猫，鹿，狗，青蛙，马，船，卡车）

CONV1_KERNEL_SIZE = 5      #第一层卷积核尺寸
CONV1_KERNEL_NUM = 6      #第一层卷积核数量
CONV2_KERNEL_SIZE = 5      #第二层卷积核尺寸
CONV2_KERNEL_NUM = 16      #第二层卷积核数量

REGULARIZER = 0.0001
BATCH_SIZE = 100
DATA_NUMS = 100000
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
STEPS = 20000
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'cifar10_conv2d_model'

def nextbatch(batch_size,dataset,datalabels,i):
    xs = dataset[i:i+batch_size]
    ys = datalabels[i:i+batch_size]
    return xs,ys


def forward(x, train, regularizer):
    #第一层卷积+池化
    conv1_b = cnn.get_bias([CONV1_KERNEL_NUM])
    conv1_w = cnn.get_weight([CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM],regularizer)
    conv1 = cnn.conv2d(x,conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
    pool1 = cnn.max_pool_2x2(relu1)
    
    #第二层卷积+池化
    conv2_b = cnn.get_bias([CONV2_KERNEL_NUM])
    conv2_w = cnn.get_weight([CONV2_KERNEL_SIZE, CONV2_KERNEL_SIZE, NUM_CHANNELS, CONV2_KERNEL_NUM],regularizer)
    conv2 = cnn.conv2d(pool1,conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))
    pool2 = cnn.max_pool_2x2(relu2)
    
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
    
    #全连接层1
    nn_w1 = cnn.get_weight([nodes, FC_SIZE], regularizer)
    nn_b1 = cnn.get_bias([FC_SIZE])
    nn_relu = tf.nn.relu(tf.matmul(reshaped,nn_w1) + nn_b1)
    if train: nn_relu = tf.nn.dropout(nn_relu, 0.5)
    
    #全连接层2
    nn_w2 = cnn.get_weight([FC_SIZE,OUTPUT_NODE],regularizer)
    nn_b2 = cnn.get_bias([OUTPUT_NODE])
    y = tf.matmul(nn_relu, nn_w2) + nn_b2
    return y

def backward(datalabels, dataimgs):
    x = tf.placeholder(tf.float32,[
        BATCH_SIZE,
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE])
    y = forward(x, True, REGULARIZER)
    global_step = tf.Variable(0, trainable = False)
    
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))
    
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        DATA_NUMS / BATCH_SIZE,#记得补上DATA_NUMS的定义
        LEARNING_RATE_DECAY,
        staircase = True
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            
        for i in range(STEPS):
            xs, ys = nextbatch(BATCH_SIZE, dataimgs, datalabels, i)
            xsa = np.reshape(xs,(BATCH_SIZE,3,32,32))
            re_xs = xsa.transpose(0,2,3,1)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: re_xs, y_: ys})
            
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g"%(step, loss_value))
                saver.save(sess. os.path.join(MODEL_SAVE_PATH,MODEL_NAME), global_step = global_step)
            
def main():
    y = np.load('newbatch_y')
    x = np.load('newbatch_x')
    backward(y, x)
    
if __name__ == "__main__":
    main()
