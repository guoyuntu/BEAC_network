import os
import numpy as np

def get_gt_theta(iter_i):
	g = open('../../result/ek2em/train/train_theta.txt', 'r')
	gtdata=g.read()
	
	gtdata = gtdata.split('\n')
	gtdata = gtdata[0:len(gtdata) - 1]
	
	ground = np.zeros([20, 2],dtype = float)
	
	num = 0
	for i in gtdata[20 *iter_i :20*iter_i +20]:
		temp = i.split(' ')
		ground[num,0] = float(temp[0])
		ground[num,1] = float(temp[1])
		num = num + 1
	
	# print('ground:' ,ground)
	return ground

def cal_mean_tIou(theta_result,iter_i):
    gt = get_gt_theta(iter_i)
    
    ground = np.zeros([20, 2],dtype = int)
    ground[:,1] = (gt[:,0] + gt[:,1] + 1)* 50
    ground[:,0] = (gt[:,1] - gt[:,0] + 1)* 50
    
    position_result = np.zeros([20, 2],dtype = int)
    position_result[:,1] = (theta_result[:,0] + theta_result[:,1] + 1)* 50
    position_result[:,0] = (theta_result[:,1] - theta_result[:,0] + 1)* 50
    # print('position_result: ',position_result)
    
    tious = np.zeros(20)
    
    b_i = np.maximum(position_result[:,0], ground[:,0])
    e_i = np.minimum(position_result[:,1], ground[:,1])
    
    b_u = np.minimum(position_result[:,0], ground[:,0])
    e_u = np.maximum(position_result[:,1], ground[:,1])
    
    tiou = np.true_divide((e_i - b_i) , (e_u - b_u))
    mean_tiou = np.mean(tiou)
    
    return mean_tiou

'''
if __name__=='__main__':
	path= '/home/g_jiarui/ideo_spacial_tu/synthetic/color/result_10/1*5_initial=0.5/'
	f = open(path + 'train_theta_result_1.txt', 'r')
	data=f.read()
	
	data = data.split('\n')
	data = data[0:len(data) - 1]
	
	pre = np.zeros((31, 2))
	
	num = 0
	
	for i in data[0:31]:
		temp = i.split(' ')
		pre[num,0] = float(temp[0])
		pre[num,1] = float(temp[1])
		num = num + 1
		
	tIou = cal_mean_tIou(pre,0)
	print(tIou)
'''
