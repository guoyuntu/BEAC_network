import read_mat
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn import metrics

x_train = read_mat.read_mat('../../data2/train_data_100.mat')['train_data'] 
y_train = read_mat.read_mat('../../data2/train_label_100.mat')['train_label'][0]

X_valid_anger = read_mat.read_mat('../../data2/validation_anger_data.mat')['validation_data']
X_valid_surprise = read_mat.read_mat('../../data2/validation_surprise_data.mat')['validation_data']
y_valid_anger = read_mat.read_mat('../../data2/validation_anger_label.mat')['validation_label'][0]
y_valid_surprise = read_mat.read_mat('../../data2/validation_surprise_label.mat')['validation_label'][0]

x_train = x_train.reshape((36000,4096))
X_valid_anger = X_valid_anger.reshape((3100,4096))
X_valid_surprise = X_valid_surprise.reshape((4900,4096))

Y_train = np.zeros((360,100))
for i in range(360):
	Y_train[i] += y_train[i]
Y_train = Y_train.reshape((36000))

x_train = preprocessing.normalize(x_train)
X_valid_anger = preprocessing.normalize(X_valid_anger)
X_valid_surprise = preprocessing.normalize(X_valid_surprise)

model = ExtraTreesClassifier()
model.fit(x_train, Y_train)
print(model.feature_importances_)

model = LinearSVC()
model.fit(x_train, Y_train)
#joblib.dump(model, 'savemodel/Ekman2.pkl')
#print(model)

p_anger = model.predict(X_valid_anger)
p_anger = p_anger.reshape((31,100))
anger = np.zeros(31)
acc_anger = 0.0
for i in range(31):
	avg = np.mean(p_anger[i])
	if avg < 0.5: 
		acc_anger += 1
	
p_surprise = model.predict(X_valid_surprise)
p_surprise = p_surprise.reshape((49,100))
surprise = np.zeros(49)
acc_surprise = 0.0
for i in range(49):
	avg = np.mean(p_surprise[i])
	if avg > 0.5: 
		acc_surprise += 1
		
print((acc_anger/31 + acc_surprise/49)/2)
print((acc_anger + acc_surprise)/80)