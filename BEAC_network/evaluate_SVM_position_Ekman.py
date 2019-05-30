import numpy as np
import os
import json
np.seterr(divide='ignore', invalid='ignore')

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def cal_mAP(url1, url2):
    f = open(url1, 'r')
    data = f.read()

    g = open(url2, 'r')
    gt = g.read()

    data = data.split('\n')
    data = data[0:len(data) - 1]
    pre = np.zeros((len(data), 2))

    num = 0
    for i in data:
        temp = i.split('   ')
        #print temp[1]
        pre[num,0] = eval(temp[1]) /2
        pre[num,1] = eval(temp[2]) /2
        num = num + 1

    gt = gt.split('\n')
    gt = gt[0:len(gt) - 1]

    ground = np.zeros((len(gt), 6))

    num = 0
    for i in gt:
        temp = i.split('   ')
        ground[num,0] = eval(temp[1])
        ground[num,1] = eval(temp[2])
        ground[num,2] = eval(temp[3])
        ground[num,3] = eval(temp[4])
        ground[num,4] = eval(temp[5])
        ground[num,5] = eval(temp[6])
        num = num + 1

    tiou = np.zeros(len(data))
    tp = np.zeros(len(data))
    fp = np.zeros(len(data))

    b_i_1 = np.maximum(pre[:,0], ground[:,0])
    e_i_1 = np.minimum(pre[:,1], ground[:,1])

    b_u_1 = np.minimum(pre[:,0], ground[:,0])
    e_u_1 = np.maximum(pre[:,1], ground[:,1])

    b_i_2 = np.maximum(pre[:,0], ground[:,2])
    e_i_2 = np.minimum(pre[:,1], ground[:,3])

    b_u_2 = np.minimum(pre[:,0], ground[:,2])
    e_u_2 = np.maximum(pre[:,1], ground[:,3])

    b_i_3 = np.maximum(pre[:,0], ground[:,4])
    e_i_3 = np.minimum(pre[:,1], ground[:,5])

    b_u_3 = np.minimum(pre[:,0], ground[:,4])
    e_u_3 = np.maximum(pre[:,1], ground[:,5])

    tiou_1 = (e_i_1 - b_i_1) / (e_u_1 - b_u_1)
    tiou_2 = (e_i_2 - b_i_2) / (e_u_2 - b_u_2)
    tiou_3 = (e_i_3 - b_i_3) / (e_u_3 - b_u_3)
    
    for i in range(len(tiou)):
        if tiou_1[i] < 0.4 and (tiou_2[i] < 0.4 or np.isnan(tiou_2[i])) and (tiou_3[i] < 0.4 or np.isnan(tiou_3[i])) :
            fp[i] = 1
        else:
            tp[i] = 1
    #tp = np.ones(3)
    tp = np.cumsum(tp).astype(np.float)
    fp = np.cumsum(fp).astype(np.float)
    rec = tp / len(data)
    prec = tp / (tp + fp)
    return interpolated_prec_rec(prec, rec)

if __name__=='__main__':

    print cal_mAP('/Volumes/Transcend/mat/Ekman/baseline/svm_position.txt','/Volumes/Transcend/mat/Ekman/baseline/position.txt');
