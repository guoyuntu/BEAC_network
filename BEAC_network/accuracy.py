import tensorflow as tf
import read_mat as read_mat
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot, weight_variable_cnn
from shuffle_data import shuffle, shuffle_emotion6, shuffle_100
from transform import transformer
from cal_tIou_Ekman6 import cal_mean_tIou, get_gt_theta
from cal_tIou_new import get_gt_position
import hdf5storage
import os
import json

mapping = {'anger':0, 'surprise':1, 'fear':2, 'joy':3, 'sadness':4, 'disgust':5}

map = json.loads(open('../../dataset/mapping.json').read())

facerate = {}
lines = open('../../result/face.txt','r').readlines()
for line in lines:
	facerate[line.split()[0]] = line.split()[1]
lines = open('../../result/face_2.txt','r').readlines()
for line in lines:
	facerate[line.split()[0]] = line.split()[1]

print('finish load data!')

x = tf.placeholder(tf.float32, [None, 100, 4096, 1])
y = tf.placeholder(tf.float32, [None, 6])
keep_prob = tf.placeholder(tf.float32)
gttheta1 = tf.placeholder(tf.float32, [None])
gttheta2 = tf.placeholder(tf.float32, [None])

x_flat = tf.reshape(x, [-1, 100 * 4096])
x_trans = tf.reshape(x, [-1, 100, 4096, 1])

W_fc_loc1 = weight_variable([100 * 4096, 20])
b_fc_loc1 = bias_variable([20])

W_fc_loc2 = weight_variable([20, 2])

initial = np.array([0.5, 0])
initial = initial.astype('float32')
initial = initial.flatten()

b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

h_fc_loc1 = tf.nn.tanh(tf.matmul(x_flat, W_fc_loc1) + b_fc_loc1)
h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

STloss = tf.reduce_mean(tf.square(gttheta1 - h_fc_loc2[:,0]) + tf.square(gttheta2 - h_fc_loc2[:,1]))
opt = tf.train.AdamOptimizer()
optimizer_ST = opt.minimize(STloss)
ST_grad = opt.compute_gradients(STloss, [b_fc_loc2])

out_size = (20, 4096)
h_trans = transformer(x_trans, h_fc_loc2, out_size)
x_trans_s = transformer(x_trans, np.array([0.5, 0.5]), out_size)

# start cnn

filter_size = 5
n_filters_1 = 8
W_conv1 = weight_variable_cnn([filter_size, 1, 1, n_filters_1])
b_conv1 = bias_variable([n_filters_1])
h_conv1 = tf.nn.relu(
	tf.nn.conv2d(input=h_trans,
				filter=W_conv1,
				strides=[1, 1, 1, 1],
				padding='SAME') +
	b_conv1)
h_conv1_2 = tf.nn.relu(
	tf.nn.conv2d(input=x_trans_s,
				filter=W_conv1,
				strides=[1, 1, 1, 1],
				padding='SAME') +
	b_conv1)

h_conv1_flat_1 = tf.reshape(h_conv1, [-1, 20*4096*n_filters_1])
h_conv1_flat_2 = tf.reshape(h_conv1_2, [-1, 20*4096*n_filters_1])
h_conv1_flat = tf.concat(1, [h_conv1_flat_1, h_conv1_flat_1])

n_fc = 32
W_fc1 = weight_variable_cnn([40 * 4096 * n_filters_1, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.tanh(tf.matmul(h_conv1_flat / 400, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable_cnn([n_fc, 6])
b_fc2 = bias_variable([6])
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
optimizer = opt.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

iter_per_epoch = 55
n_epochs = 100
train_size = 1080

indices = np.linspace(0, train_size, iter_per_epoch)
indices = indices.astype('int')
max_acc = 0
out_theta = []
out_position = []
ff = open('../../result/unsp_bi/train/accuracy.txt','w+')
print('finish building network')

saver.restore(sess, '../../model/beac_2/11.tfmodel')
acc = 0
count = 0
ff = open('../../result/face_4.txt','w+')
for i in map['test']:
	if '_'.join(i.split('_')[1:])[:-4] in facerate.keys():
		count += 1
		x_test = hdf5storage.loadmat('../../dataset/fc7_100_shrink/' + i)['feature_100']
		x_test = np.resize(x_test, (1,100,4096,1))
	
		y_test = [mapping[i.split('_')[0]]]
		y_test = dense_to_one_hot(y_test, n_classes=6)
		pred = sess.run(correct_prediction,feed_dict={x: x_test,y: y_test,keep_prob: 1.0})
	
		i = '_'.join(i.split('_')[1:])[:-4]
		if pred[0] == True:
			acc += 1
		print(i, pred[0], facerate[i])
		ff.write(i + ' ' + str(pred[0]) + ' ' + facerate[i] + '\n')
		ff.flush()
ff.close()
print(acc, count)