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

coef =4000
l2 = lambda x: x / (repmat(np.sqrt((x * x).sum(axis = 1) / coef), np.shape(x)[1], 1).T + np.finfo(np.float32).eps)

# encoding

feature = read_mat.read_mat('/home/g_jiarui/merge_fp/data/train_data_100.mat')['train_data']
position = read_mat.read_mat('/home/g_jiarui/video_spacial_tu/baseline_g/shrink_position.mat')['position']

X_valid_a = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_a_data.mat')['validation_data']
X_valid_su = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_su_data.mat')['validation_data']
X_valid_f = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_f_data.mat')['validation_data']
X_valid_j = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_j_data.mat')['validation_data']
X_valid_sa = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_sa_data.mat')['validation_data']
X_valid_d = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_d_data.mat')['validation_data']

centers = h5py.File('/home/g_jiarui/video_spacial_tu/code/mat/flickr_cluster_centers_2000.mat')['centers'][:].T
flickr = centers
norm_flickr = l2(flickr)
numcenter = np.shape(norm_flickr)[0]

center_feature = np.zeros((1080, 2000))
center_feature_a = np.zeros((31,2000))
center_feature_su = np.zeros((49,2000))
center_feature_f = np.zeros((41,2000))
center_feature_j = np.zeros((43,2000))
center_feature_sa = np.zeros((29,2000))
center_feature_ad= np.zeros((34,2000))
opt = 3000

center_feature= encoding_video_recognition(norm_flickr, feature, opt)
center_feature_a= encoding_video_recognition(norm_flickr, X_valid_a, opt)
center_feature_su= encoding_video_recognition(norm_flickr, X_valid_su, opt)
center_feature_f= encoding_video_recognition(norm_flickr, X_valid_f, opt)
center_feature_j= encoding_video_recognition(norm_flickr, X_valid_j, opt)
center_feature_sa= encoding_video_recognition(norm_flickr, X_valid_sa, opt)
center_feature_d= encoding_video_recognition(norm_flickr, X_valid_d, opt)

sio.savemat('/home/g_jiarui/merge_fp/ITE_data/center_feature.mat', {'center_feature': center_feature})
sio.savemat('/home/g_jiarui/merge_fp/ITE_data/center_feature_a.mat', {'center_feature_a': center_feature_a})
sio.savemat('/home/g_jiarui/merge_fp/ITE_data/center_feature_su.mat', {'center_feature_su': center_feature_su})
sio.savemat('/home/g_jiarui/merge_fp/ITE_data/center_feature_f.mat', {'center_feature_f': center_feature_f})
sio.savemat('/home/g_jiarui/merge_fp/ITE_data/center_feature_j.mat', {'center_feature_j': center_feature_j})
sio.savemat('/home/g_jiarui/merge_fp/ITE_data/center_feature_sa.mat', {'center_feature_sa': center_feature_sa})
sio.savemat('/home/g_jiarui/merge_fp/ITE_data/center_feature_d.mat', {'center_feature_d': center_feature_d})
'''
#calculate cos for each frame

opt = 190
feature_cos = np.zeros((np.shape(feature)[0] * np.shape(feature)[1], np.shape(fea)[0]))

num = 1
for i in range(np.shape(feature)[0]):
	for j in range(np.shape(feature)[1]):
		fea = feature[i][j]
		m = np.shape(fea)[0]
		temp = np.zeros(m)
		bow = np.zeros(m, 2000)
		weight = np.linspace(1, 0, opt)
		simi = fea * norm_flickr
		
		for wi in range(m):
			bow = np.zeros(2000)
			for d in range(len(weight)):
				y = np.max(simi[wi])
				l = np.where(simi[wi] == y)
				bow[0][l] = bow[0][l] + y * weight[d]
				simi[wi][l] = 0
			temp[wi] = np.dot(center_feature[num], bow) / (np.linalg.norm(center_feature[i]) * np.linalg.norm(bow))
		feature_cos[num] = temp
		num += 1
sio.savemat('/home/g_jiarui/merge_fp/ITE_data/feature_cos.mat', {'feature_cos': feature_cos})

#evaluate position

position_ITE = zeros(1080, 2)
for i in range(1080):
	flag = 0
	opt = np.median(feature_cos[i])
	threshold = (np.max(feature_cos[i]) - opt) / 3
	p_begin = 0
	p_end = 0
	length = 0
	max_length = 0
	endure = 0
	for j in range(np.shape(feature_cos[1]):
		if feature_cos[i][j] >= opt + threshold and flag == 0:
			p_begin = j 
			p_end = j
			flag = 1
			length = 1
		elif feature_cos[i][j] <= opt - threshold and flag == 1:
			if endure < 10:
				endure += 1
				p_end = j
				length += 1
				if length > max_length:
					max_length = length
					position_ITE[i][0] = p_begin
					position[i][1] = p_end
			else:
				flag = 0
				p_begin = 0
				p_end = 0
				length = 0
				endure = 0
		elif feature_cos[i][j] >= opt - threshold and flag == 1:
			p_end = j
			length += 1
			if length > max_length:
				max_length = length
				position_ITE[i][0] = p_begin
				position_ITE[i][1] = p_end
sio.savemat('/home/g_jiarui/merge_fp/ITE_data/position_ITE.mat', {'position_ITE': position_ITE})
'''
#train_model

x_train = read_mat.read_mat('/home/g_jiarui/merge_fp/ITE_data/center_feature.mat')['center_feature']
x_train = sparse.csr_matrix(x_train)
Y_train = read_mat.read_mat('/home/g_jiarui/merge_fp/data/train_label_100.mat')['train_label'][0]
x_train = preprocessing.normalize(x_train)
model = LinearSVC(multi_class= 'ovr', C = 60)
model.fit(x_train, Y_train)
joblib.dump(model, 'ITE_Ekman6.pkl') 
print(model)

#test

X_valid_a = read_mat.read_mat('/home/g_jiarui/merge_fp/ITE_data/center_feature_a.mat')['center_feature_a']
X_valid_a = sparse.csr_matrix(X_valid_a)
X_valid_su = read_mat.read_mat('/home/g_jiarui/merge_fp/ITE_data/center_feature_su.mat')['center_feature_su']
X_valid_su = sparse.csr_matrix(X_valid_su)
X_valid_f = read_mat.read_mat('/home/g_jiarui/merge_fp/ITE_data/center_feature_f.mat')['center_feature_f']
X_valid_f = sparse.csr_matrix(X_valid_f)
X_valid_j = read_mat.read_mat('/home/g_jiarui/merge_fp/ITE_data/center_feature_j.mat')['center_feature_j']
X_valid_j = sparse.csr_matrix(X_valid_j)
X_valid_sa = read_mat.read_mat('/home/g_jiarui/merge_fp/ITE_data/center_feature_sa.mat')['center_feature_sa']
X_valid_sa = sparse.csr_matrix(X_valid_sa)
X_valid_d = read_mat.read_mat('/home/g_jiarui/merge_fp/ITE_data/center_feature_d.mat')['center_feature_d']
X_valid_d = sparse.csr_matrix(X_valid_d)

y_valid_a = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_a_label.mat')['validation_label'][0]
y_valid_su = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_su_label.mat')['validation_label'][0]
y_valid_f = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_f_label.mat')['validation_label'][0]
y_valid_j = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_j_label.mat')['validation_label'][0]
y_valid_sa = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_sa_label.mat')['validation_label'][0]
y_valid_d = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_d_label.mat')['validation_label'][0]

X_valid_a = preprocessing.normalize(X_valid_a)
X_valid_su = preprocessing.normalize(X_valid_su)
X_valid_f = preprocessing.normalize(X_valid_f)
X_valid_j = preprocessing.normalize(X_valid_j)
X_valid_sa = preprocessing.normalize(X_valid_sa)
X_valid_d = preprocessing.normalize(X_valid_d)

model = joblib.load('ITE_Ekman6.pkl')
p_a = model.predict(X_valid_a)
acc_anger = 0.0
for i in range(31):
	if p_a[i] == y_valid_a[i] : 
		acc_anger += 1
print(acc_anger/31)

p_j = model.predict(X_valid_j)
acc_joy = 0.0
for i in range(43):
	if p_j[i] == y_valid_j[i]: 
		acc_joy += 1
print(acc_joy/43)
		
p_su = model.predict(X_valid_su)
acc_surprise = 0.0
for i in range(49):
	if p_su[i] == y_valid_su[i]: 
		acc_surprise += 1
print(acc_surprise/49)
		
p_f = model.predict(X_valid_f)
acc_fear = 0.0
for i in range(41):
	if p_f[i] == y_valid_f[i]: 
		acc_fear += 1
print(acc_fear/41)
		
p_sa = model.predict(X_valid_sa)
acc_sadness = 0.0
for i in range(29):
	if p_sa[i] == y_valid_sa[i]: 
		acc_sadness += 1
print(acc_sadness/29)

p_d = model.predict(X_valid_d)
acc_disgust = 0.0
for i in range(34):
	if p_d[i] == y_valid_d[i]: 
		acc_disgust += 1
print(acc_disgust/34)

accuracy = (acc_anger / 31 + acc_surprise / 49 + acc_fear / 41 +acc_joy / 43 + acc_sadness / 29 + acc_disgust / 34)/6
print(accuracy)