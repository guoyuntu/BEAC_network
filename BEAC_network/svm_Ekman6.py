import read_mat
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn import metrics
import scipy.stats as sis

x_train = read_mat.read_mat('/home/g_jiarui/merge_fp/data/train_data_100.mat')['train_data']
y_train = read_mat.read_mat('/home/g_jiarui/merge_fp/data/train_label_100.mat')['train_label'][0]

X_valid = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_data_100.mat')['validation_data']
y_valid = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_label_100.mat')['validation_label'][0]

X_valid_a = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_a_data.mat')['validation_data']
X_valid_su = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_su_data.mat')['validation_data']
X_valid_f = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_f_data.mat')['validation_data']
X_valid_j = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_j_data.mat')['validation_data']
X_valid_sa = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_sa_data.mat')['validation_data']
X_valid_d = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_d_data.mat')['validation_data']

y_valid_a = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_a_label.mat')['validation_label'][0]
y_valid_su = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_su_label.mat')['validation_label'][0]
y_valid_f = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_f_label.mat')['validation_label'][0]
y_valid_j = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_j_label.mat')['validation_label'][0]
y_valid_sa = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_sa_label.mat')['validation_label'][0]
y_valid_d = read_mat.read_mat('/home/g_jiarui/merge_fp/data/validation_d_label.mat')['validation_label'][0]

x_train = x_train.reshape((108000,4096))
X_valid_a = X_valid_a.reshape((3100,4096))
X_valid_su = X_valid_su.reshape((4900,4096))
X_valid_f = X_valid_f.reshape((4100,4096))
X_valid_j = X_valid_j.reshape((4300,4096))
X_valid_sa = X_valid_sa.reshape((2900,4096))
X_valid_d = X_valid_d.reshape((3400,4096))

Y_train = np.zeros((1080,100))
for i in range(1080):
	Y_train[i] += y_train[i]
Y_train = Y_train.reshape((108000))

x_train = preprocessing.normalize(x_train)
X_valid_a = preprocessing.normalize(X_valid_a)
X_valid_su = preprocessing.normalize(X_valid_su)
X_valid_f = preprocessing.normalize(X_valid_f)
X_valid_j = preprocessing.normalize(X_valid_j)
X_valid_sa = preprocessing.normalize(X_valid_sa)
X_valid_d = preprocessing.normalize(X_valid_d)

model = ExtraTreesClassifier()
model.fit(x_train, Y_train)
print(model.feature_importances_)

model = LinearSVC(multi_class = 'ovr')
model.fit(x_train, Y_train)
joblib.dump(model, 'savemodel/Ekman6.pkl') 
print(model)

model = joblib.load('savemodel/Ekman6.pkl')
p_a = model.predict(X_valid_a)
p_a = p_a.reshape((31,100))
anger = np.zeros(31)
acc_anger = 0.0
for i in range(31):
	if sis.mode(p_a[i])[0][0] == y_valid_a[i] : 
		acc_anger += 1
print(acc_anger)

p_j = model.predict(X_valid_j)
p_j = p_j.reshape((43,100))
joy = np.zeros(43)
acc_joy = 0.0
for i in range(43):
	if sis.mode(p_j[i])[0][0] == y_valid_j[i]: 
		acc_joy += 1
print(acc_joy)
		
p_su = model.predict(X_valid_su)
p_su = p_su.reshape((49,100))
surprise = np.zeros(49)
acc_surprise = 0.0
for i in range(49):
	if sis.mode(p_su[i])[0][0] == y_valid_su[i]: 
		acc_surprise += 1
print(acc_surprise)
		
p_f = model.predict(X_valid_f)
p_f = p_f.reshape((41,100))
fear = np.zeros(41)
acc_fear = 0.0
for i in range(41):
	if sis.mode(p_f[i])[0][0] == y_valid_f[i]: 
		acc_fear += 1
print(acc_fear)
		
p_sa = model.predict(X_valid_sa)
p_sa = p_sa.reshape((29,100))
sandness = np.zeros(29)
acc_sadness = 0.0
for i in range(29):
	if sis.mode(p_sa[i])[0][0] == y_valid_sa[i]: 
		acc_sadness += 1
print(acc_sadness)

p_d = model.predict(X_valid_d)
p_d = p_d.reshape((34,100))
disgust = np.zeros(34)
acc_disgust = 0.0
for i in range(34):
	if sis.mode(p_d[i])[0][0] == y_valid_d[i]: 
		acc_disgust += 1
print(acc_disgust)

accuracy = (acc_anger / 31 + acc_surprise / 49 + acc_fear / 41 +acc_joy / 43 + acc_sadness / 29 + acc_disgust / 34)/6
print(accuracy)
