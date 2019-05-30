import tensorflow as tf
import read_mat as read_mat
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot, weight_variable_cnn
from shuffle_data import shuffle, shuffle_100_ftec, shuffle_100
from transform import transformer

x_train = read_mat.read_mat('/home/g_jiarui/merge_fp/data/train_data_100.mat')['train_data'] / 400.0
y_train = read_mat.read_mat('/home/g_jiarui/merge_fp/data/train_label_100.mat')['train_label'][0]

X_test = read_mat.read_mat('/home/g_jiarui/merge_fp/data/test_data_100.mat')['test_data'] / 400.0
y_test = read_mat.read_mat('/home/g_jiarui/merge_fp/data/test_label_100.mat')['test_label'][0]

X_valid = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_data_100.mat')['validation_data'] / 400.0
y_valid = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_label_100.mat')['validation_label'][0]

X_valid_a = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_a_data.mat')['validation_data'] / 400.0
X_valid_su = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_su_data.mat')['validation_data'] / 400.0
X_valid_f = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_f_data.mat')['validation_data'] / 400.0
X_valid_j = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_j_data.mat')['validation_data'] / 400.0
X_valid_sa = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_sa_data.mat')['validation_data'] / 400.0
X_valid_d = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_d_data.mat')['validation_data'] / 400.0

y_valid_a = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_a_label.mat')['validation_label'][0]
y_valid_su = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_su_label.mat')['validation_label'][0]
y_valid_f = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_f_label.mat')['validation_label'][0]
y_valid_j = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_j_label.mat')['validation_label'][0]
y_valid_sa = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_sa_label.mat')['validation_label'][0]
y_valid_d = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_d_label.mat')['validation_label'][0]

position_train = read_mat.read_mat('/home/g_jiarui/merge_fp/data/train_position_100.mat')['train_position']
position_val = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_position_100.mat')['validation_position']
np.savetxt("/home/g_jiarui/video_spacial_tu/result/3/train/val_position.txt",position_val,fmt="%d")

#x_train_trans = np.transpose(x_train)

[X_train,y_train, position] = shuffle_100_ftec(x_train, y_train, position_train, 1080)

Y_train = dense_to_one_hot(y_train, n_classes=6)
Y_valid = dense_to_one_hot(y_valid, n_classes=6)

Y_valid_a = dense_to_one_hot(y_valid_a, n_classes=6)
Y_valid_su = dense_to_one_hot(y_valid_su, n_classes=6)
Y_valid_f = dense_to_one_hot(y_valid_f, n_classes=6)
Y_valid_j = dense_to_one_hot(y_valid_j, n_classes=6)
Y_valid_sa = dense_to_one_hot(y_valid_sa, n_classes=6)
Y_valid_d = dense_to_one_hot(y_valid_d, n_classes=6)

Y_test = dense_to_one_hot(y_test, n_classes=6)

X_train = np.resize(X_train, (1080, 100, 4096, 1))
X_valid = np.resize(X_valid, (208, 100, 4096, 1))

X_valid_a = np.resize(X_valid_a, (31, 100, 4096, 1))
X_valid_su = np.resize(X_valid_su, (49, 100, 4096, 1))
X_valid_f = np.resize(X_valid_f, (41, 100, 4096, 1))
X_valid_j = np.resize(X_valid_j, (43, 100, 4096, 1))
X_valid_sa = np.resize(X_valid_sa, (29, 100, 4096, 1))
X_valid_d = np.resize(X_valid_d, (34, 100, 4096, 1))

X_test = np.resize(X_test, (208, 100, 4096, 1))

x = tf.placeholder(tf.float32, [None, 100, 4096, 1])
y = tf.placeholder(tf.float32, [None, 6])
keep_prob = tf.placeholder(tf.float32)

x_flat = tf.reshape(x, [-1, 100 * 4096])
x_trans = tf.reshape(x, [-1, 100, 4096, 1])

W_fc_loc1 = weight_variable([100 * 4096, 20])
b_fc_loc1 = bias_variable([20])

W_fc_loc2 = weight_variable([20, 2])

initial = np.array([0.3, 0])
initial = initial.astype('float32')
initial = initial.flatten()

#b_fc_loc2 = bias_variable([6])
b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

h_fc_loc1 = tf.nn.tanh(tf.matmul(x_flat, W_fc_loc1) + b_fc_loc1)
h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

out_size = (20, 4096)
h_trans = transformer(x_trans, h_fc_loc2, out_size)

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

h_conv1_flat = tf.reshape(h_conv1, [-1, 20*4096*n_filters_1])

n_fc = 32
W_fc1 = weight_variable_cnn([20 * 4096 * n_filters_1, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.tanh(tf.matmul(h_conv1_flat / 400, W_fc1) + b_fc1)
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
grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

iter_per_epoch = 55
n_epochs = 200
train_size = 1080

indices = np.linspace(0, train_size, iter_per_epoch)
indices = indices.astype('int')
max_acc = 0;
out_theta = []
out_position = []
ff = open('/home/g_jiarui/video_spacial_tu/result/2/train/accuracy.txt','w+')

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

	acc_a = sess.run(accuracy,feed_dict={
                                                         x: X_valid_a,
                                                         y: Y_valid_a,
                                                         keep_prob: 1.0
                                                     })
	acc_su = sess.run(accuracy,feed_dict={
                                                         x: X_valid_su,
                                                         y: Y_valid_su,
                                                         keep_prob: 1.0
                                                     })
	acc_f = sess.run(accuracy,feed_dict={
                                                         x: X_valid_f,
                                                         y: Y_valid_f,
                                                         keep_prob: 1.0
                                                     })
	acc_j = sess.run(accuracy,feed_dict={
                                                         x: X_valid_j,
                                                         y: Y_valid_j,
                                                         keep_prob: 1.0
                                                     })
	acc_sa = sess.run(accuracy,feed_dict={
                                                         x: X_valid_sa,
                                                         y: Y_valid_sa,
                                                         keep_prob: 1.0
                                                     })
    acc_d = sess.run(accuracy,feed_dict={
                                                         x: X_valid_d,
                                                         y: Y_valid_d,
                                                         keep_prob: 1.0
                                                     })
    acc = str(sess.run(accuracy,feed_dict={
                                                         x: X_valid,
                                                         y: Y_valid,
                                                         keep_prob: 1.0
                                                     }))
    print('Accuracy1 (%d): ' %epoch_i + acc)
    print('Accuracy2 (%d): ' %epoch_i + str((acc_a + acc_su + acc_f + acc_j + acc_sa + acc_d) / 6))
    print('test (%d): ' % epoch_i + str(sess.run(accuracy,
                                                     feed_dict={
                                                         x: X_test,
                                                         y: Y_test,
                                                         keep_prob: 1.0
                                                     })))
    # print('y_logits (%d): ' % epoch_i + str(sess.run(W_fc1,
    #                                                  feed_dict={
    #                                                      x: X_valid,
    #                                                      y: Y_valid,
    #                                                      keep_prob: 1.0
    #                                                  })))
    
    # theta = sess.run(h_fc2_loc, feed_dict={
    #         x: batch_xs, keep_prob: 1.0})
    # print theta

    grad_vals = sess.run([g for (g,v) in grads], feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 1.0})
    print 'grad_vals: ', grad_vals

    # theta = sess.run(h_fc_loc2, feed_dict={
    #         x: batch_xs, keep_prob: 1.0})
    # print theta

    if epoch_i % 5 == 0:
    	for iter_i in range(iter_per_epoch - 1):
    		batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
    		batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]

    		theta = sess.run(h_fc_loc2, feed_dict={x: batch_xs, keep_prob: 1.0})

    		if iter_i == 0:
    			out_theta = theta
    		else:
    			out_theta = np.concatenate((out_theta,theta),axis=0)
        
    	np.savetxt("/home/g_jiarui/video_spacial_tu/result/2/train_theta_result_"+str(epoch_i)+".txt",out_theta,fmt="%f")

    	val_theta = sess.run(h_fc_loc2, feed_dict={x: X_valid, y: Y_valid, keep_prob: 1.0})
        np.savetxt("/home/g_jiarui/video_spacial_tu/result/2/val_theta_result_"+str(epoch_i)+".txt",val_theta,fmt="%f")

    	# saver.save(sess, '/home/g_jiarui/video_spacial/synthetic_data.tfmodel');
    	
    	ff.write('Accuracy_' + str(epoch_i) + '_'  + str((acc_a + acc_su + acc_f + acc_j + acc_sa + acc_d) / 6) + '\n')
    	ff.write('Accuracy_a' + str(epoch_i) + '_'  + str(acc_a) + '\n')
    	ff.write('Accuracy_su' + str(epoch_i) + '_'  + str(acc_su) + '\n')
    	ff.write('Accuracy_f' + str(epoch_i) + '_'  + str(acc_f) + '\n')
    	ff.write('Accuracy_j' + str(epoch_i) + '_'  + str(acc_j) + '\n')
    	ff.write('Accuracy_sa' + str(epoch_i) + '_'  + str(acc_sa) + '\n')
    	ff.write('Accuracy_d' + str(epoch_i) + '_'  + str(acc_d) + '\n')



    # theta = sess.run(y_logits, feed_dict={
    #     x: batch_xs, keep_prob: 1.0})
    # print theta
ff.close()

    