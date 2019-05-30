import tensorflow as tf
import read_mat as read_mat
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot, weight_variable_cnn
from shuffle_data import shuffle, shuffle_100_ftec, shuffle_100
from transform import transformer

x_train = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/train_data_100.mat')['train_data'] / 400.0
y_train = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/train_label_100.mat')['train_label'][0]

X_test = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/test_data_100.mat')['test_data'] / 400.0
y_test = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/test_label_100.mat')['test_label'][0]

X_valid = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/validation_data_100.mat')['validation_data'] / 400.0
X_valid_anger = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/validation_anger_data.mat')['validation_data'] / 400.0
X_valid_surprise = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/validation_surprise_data.mat')['validation_data'] / 400.0
y_valid_anger = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/validation_anger_label.mat')['validation_label'][0]
y_valid_surprise = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/validation_surprise_label.mat')['validation_label'][0]
y_valid = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/validation_label_100.mat')['validation_data'][0]


position_train = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/train_position_100.mat')['train_position']
position_val = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/validation_position_100.mat')['validation_position']
np.savetxt("/home/g_jiarui/video_spacial_tu/result/2/train/val_position.txt",position_val,fmt="%d")

#x_train_trans = np.transpose(x_train)

[X_train,y_train, position] = shuffle_100_cnn(x_train, y_train, position_train, 360)

Y_train = dense_to_one_hot(y_train, n_classes=2)
Y_valid_anger = dense_to_one_hot(y_valid_anger, n_classes=2)
Y_valid_surprise = dense_to_one_hot(y_valid_surprise, n_classes=2)
Y_valid = dense_to_one_hot(y_valid, n_classes=2)
Y_test = dense_to_one_hot(y_test, n_classes=2)

X_train = np.resize(X_train, (360, 100, 4096, 1))
X_valid = np.resize(X_valid, (80, 100, 4096, 1))
X_valid_anger = np.resize(X_valid_anger, (31, 100, 4096, 1))
X_valid_surprise = np.resize(X_valid_surprise, (49, 100, 4096, 1))
X_test = np.resize(X_test, (80, 100, 4096, 1))

x = tf.placeholder(tf.float32, [None, 100, 4096, 1])
y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)

x_flat = tf.reshape(x, [-1, 100 * 4096])
x_trans = tf.reshape(x, [-1, 100, 4096, 1])

#b_fc_loc2 = bias_variable([6])

# start cnn

filter_size = 5
n_filters_1 = 8
W_conv1 = weight_variable_cnn([filter_size, 1, 1, n_filters_1])
b_conv1 = bias_variable([n_filters_1])
h_conv1 = tf.nn.relu(
	tf.nn.conv2d(input=x,
				filter=W_conv1,
				strides=[1, 1, 1, 1],
				padding='SAME') +
	b_conv1)

h_conv1_flat = tf.reshape(h_conv1, [-1, 100*4096*n_filters_1])

n_fc = 32
W_fc1 = weight_variable_cnn([100 * 4096 * n_filters_1, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.tanh(tf.matmul(h_conv1_flat / 400, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable_cnn([n_fc, 2])
b_fc2 = bias_variable([2])
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
opt = tf.train.AdamOptimizer()
optimizer = opt.minimize(cross_entropy)

correct_prediction_anger = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy_anger = tf.reduce_mean(tf.cast(correct_prediction_anger, 'float'))
correct_prediction_surprise = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy_surprise = tf.reduce_mean(tf.cast(correct_prediction_surprise, 'float'))
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

iter_per_epoch = 19
n_epochs = 200
train_size = 360

indices = np.linspace(0, train_size, iter_per_epoch)
indices = indices.astype('int')
max_acc = 0;
out_theta = []
out_position = []
ff = open('/home/g_jiarui/video_spacial_tu/result/4/train/accuracy.txt','w+')

for epoch_i in range(n_epochs):
    for iter_i in range(iter_per_epoch - 1):
        batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
        batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]
        		     
        if iter_i % 10 == 0:
            loss = sess.run(cross_entropy,
                            feed_dict={
                                x: batch_xs,
                                y: batch_ys,
                                keep_prob: 1.0
                            })
            print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))

        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.75})

    acc_anger = sess.run(accuracy_anger,feed_dict={
                                                         x: X_valid_anger,
                                                         y: Y_valid_anger,
                                                         keep_prob: 1.0
                                                     })
    acc_surprise = sess.run(accuracy_surprise,feed_dict={
                                                         x: X_valid_surprise,
                                                         y: Y_valid_surprise,
                                                         keep_prob: 1.0
                                                     })                                                    
    print('Accuracy_anger (%d): ' %epoch_i + str(acc_anger))
    print('Accuracy_asurprise (%d): ' %epoch_i + str(acc_surprise))
    # print('y_logits (%d): ' % epoch_i + str(sess.run(W_fc1,
    #                                                  feed_dict={
    #                                                      x: X_valid,
    #                                                      y: Y_valid,
    #                                                      keep_prob: 1.0
    #                                                  })))
    
    # theta = sess.run(h_fc2_loc, feed_dict={
    #         x: batch_xs, keep_prob: 1.0})
    # print theta


    # theta = sess.run(h_fc_loc2, feed_dict={
    #         x: batch_xs, keep_prob: 1.0})
    # print theta

    if epoch_i % 5 == 0:
    	ff.write('Accuracy_' + str(epoch_i) + '_'  + str((acc_anger + acc_surprise) / 2) + '\n')



    # theta = sess.run(y_logits, feed_dict={
    #     x: batch_xs, keep_prob: 1.0})
    # print theta
ff.close()

    