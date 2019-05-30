import read_mat as read_mat
import numpy as np
import cv2
import scipy.io as sio
import hdf5storage

ek = read_mat.read_mat('../../data/test_data_100.mat')['test_data']
em = hdf5storage.loadmat('../../data/multi_test_data.mat')['test_data']

def ali(ds1, ds2, al_1=True):
		print(ds1.shape)
		print(ds2.shape)
		ds1 = np.resize(ds1, (208, 100, 4096))
		ds2 = np.resize(ds2, (600, 30, 4096))
		if al_1:
			a_ds1 = np.zeros((208,30,4096))
			for i in range(208):
				pic = ds1[i]
				pic = cv2.resize(pic, (4096, 30), interpolation=cv2.INTER_LINEAR)
				a_ds1[i] = pic
			sio.savemat('../../data/test_data_30.mat', {'test_data': a_ds1})
			print('finsih convert ekman6', a_ds1.shape)
			
		if not al_1:
			a_ds1 = np.zeros((600,100,4096))
			for i in range(600):
				pic = ds2[i]
				pic = cv2.resize(pic, (4096, 100), interpolation=cv2.INTER_LINEAR)
				a_ds1[i] = pic
			sio.savemat('../../data/multi_test_data_100.mat', {'test_data': a_ds1})
			print('finish convert emotion6', a_ds1.shape)
			
ali(ek, em, True)
ali(ek, em, False)