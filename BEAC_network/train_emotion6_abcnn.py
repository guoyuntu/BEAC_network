import tensorflow as tf
import read_mat as read_mat
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot, weight_variable_cnn
from shuffle_data import shuffle, shuffle_emotion6, shuffle_100
from cal_tIou_Ekman6 import cal_mean_tIou, get_gt_theta
from cal_tIou_new import get_gt_position
import hdf5storage
from sklearn.metrics import classification_report,recall_score

x_train = hdf5storage.loadmat('../../data/multi_train_data.mat')['train_data']
y_train = hdf5storage.loadmat('../../data/multi_train_label.mat')['train_label'][0]
position_train = hdf5storage.loadmat('../../data/multi_train_position.mat')['train_position']

[X_train,y_train, position] = shuffle_emotion6(x_train, y_train, position_train, 2400)

Y_train = dense_to_one_hot(y_train, n_classes=6)

x_valid = hdf5storage.loadmat('../../data/multi_validation_data.mat')['validation_data']
y_valid = hdf5storage.loadmat('../../data/multi_validation_label.mat')['validation_label'][0]

Y_valid = dense_to_one_hot(y_valid, n_classes=6)

x_test = hdf5storage.loadmat('../../data/multi_test_data.mat')['test_data']
y_test = hdf5storage.loadmat('../../data/multi_test_label.mat')['test_label'][0]

Y_test = dense_to_one_hot(y_test, n_classes=6)

position_test = hdf5storage.loadmat('../../data/multi_test_position.mat')['test_position']

X_train = np.resize(X_train, (2400, 30, 4096, 1))
X_valid = np.resize(x_valid, (600, 30, 4096, 1))
X_test = np.resize(x_test, (600, 30, 4096, 1))

np.savetxt("../../result/att_em/train/val_position.txt",position_test,fmt="%d")

print('finish load data!')

x = tf.placeholder(tf.float32, [None, 30, 4096, 1])
y = tf.placeholder(tf.float32, [None, 6])
keep_prob = tf.placeholder(tf.float32)

#soft attention
x_flat = tf.reshape(x, [-1, 30 * 4096])
x_trans = tf.reshape(x, [-1, 30, 4096])

W_fc_loc1 = weight_variable([30 * 4096, 30])
b_fc_loc1 = bias_variable([30])

alpha = tf.nn.softmax(tf.matmul(x_flat, W_fc_loc1) + b_fc_loc1)

alpha_tensor = tf.tile(tf.expand_dims(alpha, -1), [1,1,4096])
attention_output = tf.multiply(x_trans, alpha_tensor)
attention_output = tf.expand_dims(attention_output, -1)

# start cnn

filter_size = 5
n_filters_1 = 8
W_conv1 = weight_variable_cnn([filter_size, 1, 1, n_filters_1])
b_conv1 = bias_variable([n_filters_1])
h_conv1 = tf.nn.relu(
	tf.nn.conv2d(input=attention_output,
				filter=W_conv1,
				strides=[1, 1, 1, 1],
				padding='SAME') +
	b_conv1)

h_conv1_flat = tf.reshape(h_conv1, [-1, 30*4096*n_filters_1])

n_fc = 32
W_fc1 = weight_variable_cnn([30 * 4096 * n_filters_1, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.tanh(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable_cnn([n_fc, 6])
b_fc2 = bias_variable([6])
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
opt = tf.train.AdamOptimizer()
optimizer = opt.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

iter_per_epoch = 61
n_epochs = 30
train_size = 2400

indices = np.linspace(0, train_size, iter_per_epoch)
indices = indices.astype('int')
max_acc = 0
out_theta = []
out_position = []
ff = open('../../result/att_em/train/accuracy.txt','w+')
print('finish building network')
#target_names = ['anger', 'suprise', 'fear', 'joy', 'sad', 'disgust']

#saver.restore(sess, '../../model/unsp_bi_em/5.tfmodel')

for epoch_i in range(n_epochs):
    for iter_i in range(iter_per_epoch - 1):
        batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
        batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]

        if iter_i % 10 == 0:
            loss = sess.run(cross_entropy,feed_dict={x: batch_xs,y: batch_ys,keep_prob: 1.0})
            train_acc = str(sess.run(accuracy,feed_dict={x: batch_xs,y: batch_ys,keep_prob: 1.0}))
            print('Epoch: ' + str(epoch_i) + ' Iteration: ' + str(iter_i) + ' Loss: ' + str(loss) + ' Train acc: '+ str(train_acc))

        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.75})
    
    if epoch_i % 1 == 0:
        pred_valid = sess.run(tf.argmax(y_logits ,1),feed_dict={x: X_valid,y: Y_valid,keep_prob: 1.0})
        pred = sess.run(tf.argmax(y_logits, 1),feed_dict={x: X_test,y: Y_test,keep_prob: 1.0})
        
        acc_valid = recall_score(y_valid, pred_valid, average='macro')
        acc = recall_score(y_test, pred, average='macro')

        print('valid (%d): ' % epoch_i + str(acc_valid))
        print('test (%d): ' % epoch_i + str(acc))
        
        print(classification_report(y_valid, pred_valid))
        
        saver.save(sess, '../../model/att_em/' + str(epoch_i) + '.tfmodel');
        
        ff.write('Accuracy_valid_' + str(epoch_i) + '_'  + str(acc_valid) + ' Accuracy_test_' + str(epoch_i) + '_'  + str(acc) + '\n')
        ff.flush()

ff.close()	
