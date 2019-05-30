import numpy as np
import json
import read_mat as read_mat
import os
import scipy.io as sio 
import random
'''
def process_attribution():
	path = '/Users/appletgj/Desktop/fc7_100_shrink/'
	filelist =  os.listdir(path)
	f = open('attributions.json')
	at = json.loads(f.read())
	dic = {'anger':0,'surprise':0,'fear':0,'joy':0,'sadness':0,'disgust':0}
	for i in filelist:
		if i.startswith('anger'):dic['anger'] += 1
		if i.startswith('surprise'):dic['surprise'] += 1
		if i.startswith('fear'):dic['fear'] += 1
		if i.startswith('joy'):dic['joy'] += 1
		if i.startswith('sadness'):dic['sadness'] += 1
		if i.startswith('disgust'):dic['disgust'] += 1
	print(dic)
process_attribution()
'''

def split_data():
	frame = 100
	opt = 0.15

	num = np.array([[213,331,276,289,195,234]])
	num_opt = (num * opt).astype(int)
	
	path = '/Users/appletgj/Desktop/fc7_100_shrink/'
	filelist =  os.listdir(path)[1:-1]
	f = open('attributions.json')
	at = json.loads(f.read())
	
	dic = {'anger':list(),'surprise':list(),'fear':list(),'joy':list(),'sadness':list(),'disgust':list()}
	
	for i in filelist:
		if i.startswith('anger'):dic['anger'].append(i.split('.mp4.mat')[0])
		if i.startswith('surprise'):dic['surprise'].append(i.split('.mp4.mat')[0])
		if i.startswith('fear'):dic['fear'].append(i.split('.mp4.mat')[0])
		if i.startswith('joy'):dic['joy'].append(i.split('.mp4.mat')[0])
		if i.startswith('sadness'):dic['sadness'].append(i.split('.mp4.mat')[0])
		if i.startswith('disgust'):dic['disgust'].append(i.split('.mp4.mat')[0])
	
	category = ['anger', 'surprise', 'fear', 'joy', 'sadness', 'disgust']
	
	for i in category:
		dic[i] = random.sample(dic[i],len(dic[i]))
	
	
	dic2 = {'validation':list(),'test':list(),'train':list()}
	
	for j in range(6):
		for i in range(num[0,j]):
			if i < num_opt[0,j]:
				dic2['validation'].append(dic[category[j]][i] + '.mp4.mat')
			elif i < 2 * num_opt[0,j]:
				dic2['test'].append(dic[category[j]][i]+ '.mp4.mat')
			else:
				dic2['train'].append(dic[category[j]][i]+ '.mp4.mat')	
	
	ff = open('/Users/appletgj/Desktop/video_spacial_tu/mapping.json', 'w')
	ff.write(json.dumps(dic2))
	ff.close()

def merge_f_p():
	frame = 100
	
	test_data = np.zeros((208, frame, 4096))
	validation_data = np.zeros((208, frame, 4096))
	train_data = np.zeros((1082, frame, 4096))
	
	test_label = np.zeros((1, 208), dtype = np.int)
	validation_label = np.zeros((1, 208), dtype = np.int)
	train_label = np.zeros((1, 1082), dtype = np.int)
	
	test_position = np.zeros((208, 2), dtype = np.int)
	validation_position = np.zeros((208, 2), dtype = np.int)
	train_position = np.zeros((1082, 2), dtype = np.int)
	
	path = '/Users/appletgj/Desktop/fc7_100_shrink/'
	f = open('attributions.json')
	at = json.loads(f.read())
	map = json.loads(open('mapping.json').read())
	
	for i in range(208):
		shape = np.shape(read_mat.read_mat(path + map['validation'][i])['feature_100'])
		validation_data[i,:shape[0],:] = read_mat.read_mat(path + map['validation'][i])['feature_100']
		if map['validation'][i].startswith('anger'):validation_label[0, i] = 0
		if map['validation'][i].startswith('surprise'):validation_label[0, i] = 1
		if map['validation'][i].startswith('fear'):validation_label[0, i] = 2
		if map['validation'][i].startswith('joy'):validation_label[0, i] = 3
		if map['validation'][i].startswith('sadness'):validation_label[0, i] = 4
		if map['validation'][i].startswith('disgust'):validation_label[0, i] = 5
		validation_position[i,0] = int(100 * int(at[map['validation'][i]]['clip'].split('-')[0]) / int(at[map['validation'][i]]['length']))
		validation_position[i,1] = int(100 * int(at[map['validation'][i]]['clip'].split('-')[1]) / int(at[map['validation'][i]]['length']))
	'''
	for i in range(208):
		shape = np.shape(read_mat.read_mat(path + map['test'][i])['feature_100'])
		test_data[i,:shape[0],:] = read_mat.read_mat(path + map['test'][i])['feature_100']
		if map['test'][i].startswith('anger'):test_label[0, i] = 0
		if map['test'][i].startswith('surprise'):test_label[0, i] = 1
		if map['test'][i].startswith('fear'):test_label[0, i] = 2
		if map['test'][i].startswith('joy'):test_label[0, i] = 3
		if map['test'][i].startswith('sadness'):test_label[0, i] = 4
		if map['test'][i].startswith('disgust'):test_label[0, i] = 5
		test_position[i,0] = int(100 * int(at[map['test'][i]]['clip'].split('-')[0]) / int(at[map['test'][i]]['length']))
		test_position[i,1] = int(100 * int(at[map['test'][i]]['clip'].split('-')[1]) / int(at[map['test'][i]]['length']))
	
	for i in range(1082):
		shape = np.shape(read_mat.read_mat(path + map['train'][i])['feature_100'])
		train_data[i,:shape[0],:] = read_mat.read_mat(path + map['train'][i])['feature_100']
		if map['train'][i].startswith('anger'):train_label[0, i] = 0
		if map['train'][i].startswith('surprise'):train_label[0, i] = 1
		if map['train'][i].startswith('fear'):train_label[0, i] = 2
		if map['train'][i].startswith('joy'):train_label[0, i] = 3
		if map['train'][i].startswith('sadness'):train_label[0, i] = 4
		if map['train'][i].startswith('disgust'):train_label[0, i] = 5
		train_position[i,0] = int(100 * int(at[map['train'][i]]['clip'].split('-')[0]) / int(at[map['train'][i]]['length']))
		train_position[i,1] = int(100 * int(at[map['train'][i]]['clip'].split('-')[1]) / int(at[map['train'][i]]['length']))
	'''
	#sio.savemat('/users/appletgj/desktop/data/train_data_100.mat', {'train_data': train_data})
	#sio.savemat('/users/appletgj/desktop/data/test_data_100.mat', {'test_data': test_data})
	sio.savemat('/users/appletgj/desktop/data/validation_data_100.mat', {'validation_data': validation_data})
	
	#sio.savemat('/users/appletgj/desktop/data/train_label_100.mat', {'train_label': train_label})
	#sio.savemat('/users/appletgj/desktop/data/test_label_100.mat', {'test_label': test_label})
	sio.savemat('/users/appletgj/desktop/data/validation_label_100.mat', {'validation_label': validation_label})
	
	#sio.savemat('/users/appletgj/desktop/data/train_position_100.mat', {'train_position': train_position})
	#sio.savemat('/users/appletgj/desktop/data/test_position_100.mat', {'test_position': test_position})
	sio.savemat('/users/appletgj/desktop/data/validation_position_100.mat', {'validation_position': validation_position})

#split_data()
merge_f_p()


