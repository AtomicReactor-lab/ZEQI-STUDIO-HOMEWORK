import tensorflow as tf
import numpy as np
import os
import pickle

CIFAR_DIR = "./cifar-10-batches-py"


# 指定使用哪块gpu，如果没有，则注释掉
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
# 数据载入函数
def load_data(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f,encoding='bytes')
        return data[b'data'],data[b'labels']


# 数据处理函数
class CifarDate:
    def __init__(self,filenames,need_shuffle):
        all_data = []
        all_label = []
        for filename in filenames:
            data,labels = load_data(filename)
            all_data.append(data)
            all_label.append(labels)
        self._data = np.vstack(all_data)
        self._labels = np.hstack(all_label)

        self.start = 0
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        if self._need_shuffle:
            self._shuffle_data()
	
	# 洗牌
    def _shuffle_data(self):
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]
	
	# 小批量数据集获取
    def next_batch(self,batch_size):
        end = self.start + batch_size
        if end > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self.start = 0
                end = batch_size
            else:
                raise Exception('have no more examples')
        if end > self._num_examples:
            raise Exception('batch size is larger than all examplts')

        batch_data = self._data[self.start:end]
        batch_labels = self._labels[self.start:end]
        self.start = end
        return batch_data,batch_labels


train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]

# 实例化数据处理对象
train_data = CifarDate(train_filenames, True)
test_data = CifarDate(test_filenames, False)

# 批量
batch_sezi = 40
# 占位符
x = tf.placeholder(tf.float32,[None,3072])
y = tf.placeholder(tf.int64,[None])

# 转成图片格式
x_image = tf.reshape(x,[-1,3,32,32])
# 转成卷积网络适应的格式
x_image = tf.transpose(x_image,perm=[0,2,3,1])

# 将一批图像切分成一张一张的，组成一个数组
x_image_arr = tf.split(x_image,num_or_size_splits=batch_sezi,axis=0)
result_x_iamge_arr = []

# 数据增广
for x_single_image in x_image_arr:
    x_single_image = tf.reshape(x_single_image,[32,32,3])

    # 随机反转
    data_aug_1 = tf.image.random_flip_left_right(x_single_image)

    # 调整光照
    data_aug_2 = tf.image.random_brightness(data_aug_1,max_delta=63)

    # 改变对比度
    data_aug_3 = tf.image.random_contrast(data_aug_2,lower=0.2,upper=1.8)

    # 白化
    data_aug_4 = tf.image.per_image_standardization(data_aug_3)
    # 标准化
    data_aug_4 = data_aug_4 / 127.5 - 1
    data_aug_3 = data_aug_3 / 127.5 - 1
    data_aug_2 = data_aug_2 / 127.5 - 1
    data_aug_1 = data_aug_1 / 127.5 - 1
    x_single_image  = x_single_image / 127.5 - 1

    x_single_image = tf.reshape(x_single_image,[1,32,32,3])
    data_aug_4 = tf.reshape(data_aug_4,[1,32,32,3])
    data_aug_3 = tf.reshape(data_aug_3,[1,32,32,3])
    data_aug_2 = tf.reshape(data_aug_2,[1,32,32,3]) 
    data_aug_1 = tf.reshape(data_aug_1,[1,32,32,3])
	
	# 将每个增广过程的图片都添加为训练图片
    result_x_iamge_arr.append(x_single_image)
    result_x_iamge_arr.append(data_aug_4)
    result_x_iamge_arr.append(data_aug_3)
    result_x_iamge_arr.append(data_aug_2)
    result_x_iamge_arr.append(data_aug_1)


# 增广后的数据合并
normal_result_x_image = tf.concat(result_x_iamge_arr,axis=0)


# 卷积函数
def conv_wrapper(inputs,name,is_training,output_channel,kernel_size=(3,3),
                 activation=tf.nn.relu,padding='same'):
    conv2d = tf.layers.conv2d(inputs,output_channel,kernel_size,padding=padding,name=name + '/conv2d',
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    bn = tf.layers.batch_normalization(conv2d,training=is_training)
    return activation(bn)


# 池化函数
def pooling_wrapper(inputs,name):
    return tf.layers.max_pooling2d(inputs,(2,2),(2,2),name=name)


# 卷积一
conv1_1 = conv_wrapper(normal_result_x_image,'conv1_1',True,64)
conv1_2 = conv_wrapper(conv1_1,'conv1_2',True,64)
conv1_3 = conv_wrapper(conv1_2,'conv1_3',True,64)
pooling1 = pooling_wrapper(conv1_3,'pool1')

# 卷积二
conv2_1 = conv_wrapper(pooling1,'conv2_1',True,128)
conv2_2 = conv_wrapper(conv1_1,'conv2_2',True,128)
conv2_3 = conv_wrapper(conv1_2,'conv2_3',True,128)
pooling2 = pooling_wrapper(conv2_3,'pool2')

# 卷积三
conv3_1 = conv_wrapper(pooling2,'conv3_1',True,256)
conv3_2 = conv_wrapper(conv3_1,'conv3_2',True,256)
conv3_3 = conv_wrapper(conv3_2,'conv3_3',True,256)
pooling3 = pooling_wrapper(conv3_3,'pool3')

# 展平
flatten = tf.layers.flatten(pooling3)

fc1 = tf.layers.dense(flatten,512,activation=tf.nn.relu)
fc2 = tf.layers.dense(fc1,256,activation=tf.nn.relu)
fc3 = tf.layers.dense(fc2,128,activation=tf.nn.relu)

y_ = tf.layers.dense(fc3,10)

# 代价
loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)

# 预测
predict = tf.argmax(y_,1)

# 准确率
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,y),dtype=tf.float32))

# 优化器
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

# 将训练过程中的参数模型保存在本地
LOG_DIR = '.'
run_label = 'run_cgg_tensorboard'
run_dir = os.path.join(LOG_DIR,run_label)

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

train_log_dir = os.path.join(run_dir,'train')
test_log_dir = os.path.join(run_dir,'test')

if not os.path.exists(train_log_dir):
    os.mkdir(train_log_dir)
if not os.path.exists(test_log_dir):
    os.mkdir(test_log_dir)

# data文件存储参数数据，index文件存储索引信息，meta文件存储源信息
model_dir = os.path.join(run_dir,'model')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

saver = tf.train.Saver()
model_name = 'ckp-20000'
model_path = os.path.join(model_dir,model_name)

out_put_every_steps = 500
train_steps = 20000
test_steps = 100

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if os.path.exists(model_path+'.index'):
        saver.restore(sess,model_path)  #model_path存储的参数初始化sess
        print('model restored from %s' % model_path)
    else:
        print('model %s does not exist' % model_path)

    for i in range(train_steps):
        batch_data,batch_labels = train_data.next_batch(batch_sezi)
		# 标签增广
        batch_labels = [s for s in batch_labels for l in range(5)]
        # batch_labels = sess.run(tf.reshape(batch_labels,[-1,1]))
        loss_val, acc_val, _ = sess.run([loss,accuracy,train_op],feed_dict={
            x:batch_data,y:batch_labels
        })

        if (i+1) % 500 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f'
                  % (i + 1, loss_val, acc_val))

        if (i+1) % 5000 == 0:
            test_data = CifarDate(test_filenames,False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels = test_data.next_batch(batch_sezi)
                test_batch_labels = [s for s in test_batch_labels for l in range(5)]
               	
                test_acc_val = sess.run(accuracy,feed_dict={
                    x:test_batch_data,y:test_batch_labels
                })
                all_test_acc_val.append(test_acc_val)
            test_acc = sess.run(tf.reduce_mean(all_test_acc_val))
            print('测试集准确率',test_acc)

        if (i+1) %  out_put_every_steps == 0:
            saver.save(sess,os.path.join(model_dir,'ckp-%05d' % (i+1)))
            print('model saved to ckp-%05d' % (i+1))
