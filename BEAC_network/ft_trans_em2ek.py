import tensorflow as tf
import read_mat as read_mat
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot, weight_variable_cnn
from shuffle_data import shuffle, shuffle_emotion6, shuffle_100
from transform import transformer
from cal_tIou_Ekman6 import cal_mean_tIou, get_gt_theta
from cal_tIou_new import get_gt_position
import hdf5storage

x_train = hdf5storage.loadmat('../../data/train_data_30.mat')['train_data'] / 400.0
y_train = hdf5storage.loadmat('../../data/train_label_100.mat')['train_label'][0]
position_train = read_mat.read_mat('../../data/train_position_100.mat')['train_position']

[X_train,y_train, position] = shuffle_emotion6(x_train, y_train, position_train, 1080)

X_test = X_train[-208:]
y_test = y_train[-208:]

X_train = X_train[:220]
y_train = y_train[:220]
position = position[:220]

Y_train = dense_to_one_hot(y_train, n_classes=6)
Y_test = dense_to_one_hot(y_test, n_classes=6)

X_train = np.resize(X_train, (220, 30, 4096, 1))
X_test = np.resize(X_test, (208, 30, 4096, 1))

print('finish load data!')

x = tf.placeholder(tf.float32, [None, 30, 4096, 1])
y = tf.placeholder(tf.float32, [None, 6])
keep_prob = tf.placeholder(tf.float32)
gttheta1 = tf.placeholder(tf.float32, [None])
gttheta2 = tf.placeholder(tf.float32, [None])

x_flat = tf.reshape(x, [-1, 30 * 4096])
x_trans = tf.reshape(x, [-1, 30, 4096, 1])

W_fc_loc1 = weight_variable([30 * 4096, 20])
b_fc_loc1 = bias_variable([20])

W_fc_loc2 = weight_variable([20, 2])

initial = np.array([0.5, 0])
initial = initial.astype('float32')
initial = initial.flatten()

#b_fc_loc2 = bias_variable([6])
b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

h_fc_loc1 = tf.nn.relu(tf.matmul(x_flat, W_fc_loc1) + b_fc_loc1)
h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
h_fc_loc2 = tf.nn.relu(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

STloss = tf.reduce_mean(tf.square(gttheta1 - h_fc_loc2[:,0]) + tf.square(gttheta2 - h_fc_loc2[:,1]))
opt = tf.train.AdamOptimizer()
optimizer_ST = opt.minimize(STloss)
ST_grad = opt.compute_gradients(STloss, [b_fc_loc2])

out_size = (20, 4096)
h_trans = transformer(x_trans, h_fc_loc2, out_size)
x_trans_s = transformer(x_trans, np.array([0.5, 0.5]), out_size)

#h_multi = tf.concat(1, [h_trans, x_trans])

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
h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat / 400, W_fc1) + b_fc1)
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

iter_per_epoch = 12
n_epochs = 10
train_size = 220

indices = np.linspace(0, train_size, iter_per_epoch)
indices = indices.astype('int')
max_acc = 0
out_theta = []
out_position = []
print('finish building network')

#saver.restore(sess, '../../model/beac/11.tfmodel')

#fine-tuning
for epoch_i in range(n_epochs):
    for iter_i in range(iter_per_epoch - 1):
        batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
        batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]
        
        '''STloss is backpropagated through the STN'''
        threshold = 0.3
        theta = sess.run(h_fc_loc2, feed_dict={x: batch_xs, keep_prob: 1.0})
        if epoch_i > 10:
        	if cal_mean_tIou(theta,iter_i) < threshold:
        		ST_loss = sess.run(STloss,feed_dict={x: batch_xs,gttheta1: get_gt_theta(iter_i)[:,0],gttheta2: get_gt_theta(iter_i)[:,1],keep_prob: 1.0})
        		if iter_i % 5 == 0:
        			print('epoch_i: ' + str(epoch_i) + ' iter_i: ' + str(iter_i) +' ST_loss: ' + str(ST_loss))
        		sess.run(optimizer_ST,feed_dict={x: batch_xs,gttheta1: get_gt_theta(iter_i)[:,0],gttheta2: get_gt_theta(iter_i)[:,1],keep_prob:0.75})
        		continue

        if iter_i % 5 == 0:
        	#print('mean_tIou: ',cal_mean_tIou(theta,iter_i))
                loss = sess.run(cross_entropy,feed_dict={x: batch_xs,y: batch_ys,keep_prob: 1.0})
                train_acc = str(sess.run(accuracy,feed_dict={x: batch_xs,y: batch_ys,keep_prob: 1.0}))
                print('Epoch: ' + str(epoch_i) + ' Iteration: ' + str(iter_i) + ' Loss: ' + str(loss) + ' Train acc: '+ str(train_acc))

        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.75})

    acc = sess.run(accuracy,feed_dict={x: X_test,y: Y_test,keep_prob: 1.0})
    print('test (%d): %.4f' %(epoch_i, acc))
    #finetuning: epoch3 15.87% epoch5 21.15% epoch10 21.63%
    #without: epoch3 16.50% epoch5 17.50% epoch10 18.00%
