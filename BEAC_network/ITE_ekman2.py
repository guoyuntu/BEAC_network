from encoding_video_recognition import *
import numpy as np
import read_mat
import h5py
import scipy.io as sio 
from scipy import sparse
from numpy.matlib import repmat
from sklearn.externals import joblib
from sklearn import *
from sklearn.svm import *
'''
coef =4000
l2 = lambda x: x / (repmat(np.sqrt((x * x).sum(axis = 1) / coef), np.shape(x)[1], 1).T + np.finfo(np.float32).eps)

# encoding

feature = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/train_data_100.mat')['train_data']

X_valid_a = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/validation_anger_data.mat')['validation_data']
X_valid_su = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/validation_surprise_data.mat')['validation_data']

centers = h5py.File('/home/g_jiarui/video_spacial_tu/code/mat/flickr_cluster_centers_2000.mat')['centers'][:].T
flickr = centers
norm_flickr = l2(flickr)
numcenter = np.shape(norm_flickr)[0]

center_feature = np.zeros((360, 2000))
center_feature_a = np.zeros((31,2000))
center_feature_su = np.zeros((49,2000))
opt = 2000

center_feature= encoding_video_recognition(norm_flickr, feature, opt)
center_feature_a= encoding_video_recognition(norm_flickr, X_valid_a, opt)
center_feature_su= encoding_video_recognition(norm_flickr, X_valid_su, opt)

sio.savemat('/home/g_jiarui/merge_fp/ITE_data2/center_feature.mat', {'center_feature': center_feature})
sio.savemat('/home/g_jiarui/merge_fp/ITE_data2/center_feature_a.mat', {'center_feature_a': center_feature_a})
sio.savemat('/home/g_jiarui/merge_fp/ITE_data2/center_feature_su.mat', {'center_feature_su': center_feature_su})
'''

#train_model

x_train = read_mat.read_mat('/home/g_jiarui/merge_fp/ITE_data2/center_feature.mat')['center_feature']
x_train = sparse.csr_matrix(x_train)
Y_train = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/train_label_100.mat')['train_label'][0]
x_train = preprocessing.normalize(x_train)
model = LinearSVC(C = 60)
model.fit(x_train, Y_train)
joblib.dump(model, 'ITE_Ekman2.pkl') 
print(model)

#test

X_valid_a = read_mat.read_mat('/home/g_jiarui/merge_fp/ITE_data2/center_feature_a.mat')['center_feature_a']
X_valid_a = sparse.csr_matrix(X_valid_a)
X_valid_su = read_mat.read_mat('/home/g_jiarui/merge_fp/ITE_data2/center_feature_su.mat')['center_feature_su']
X_valid_su = sparse.csr_matrix(X_valid_su)

y_valid_a = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/validation_anger_label.mat')['validation_label'][0]
y_valid_su = read_mat.read_mat('/home/g_jiarui/merge_fp/data2/validation_surprise_label.mat')['validation_label'][0]

X_valid_a = preprocessing.normalize(X_valid_a)
X_valid_su = preprocessing.normalize(X_valid_su)

model = joblib.load('ITE_Ekman2.pkl')
p_a = model.predict(X_valid_a)
acc_anger = 0.0
for i in range(31):
	if p_a[i] == y_valid_a[i] : 
		acc_anger += 1
print(acc_anger/31)
		
p_su = model.predict(X_valid_su)
acc_surprise = 0.0
for i in range(49):
	if p_su[i] == y_valid_su[i]: 
		acc_surprise += 1
print(acc_surprise/49)

accuracy = (acc_anger / 31 + acc_surprise / 49)/2
print(accuracy)