import numpy as np
import os
import json

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
        pre[num,0] = eval(temp[1])
        pre[num,1] = eval(temp[2])
        num = num + 1

    gt = gt.split('\n')
    gt = gt[0:len(gt) - 1]

    ground = np.zeros((len(gt), 2))

    num = 0
    for i in gt:
        temp = i.split('   ')
        ground[num,0] = eval(temp[1])
        ground[num,1] = eval(temp[2])
        num = num + 1

    tiou = np.zeros(len(data))
    tp = np.zeros(len(data))
    fp = np.zeros(len(data))

    b_i = np.maximum(pre[:,0], ground[:,0])
    e_i = np.minimum(pre[:,1], ground[:,1])

    b_u = np.minimum(pre[:,0], ground[:,0])
    e_u = np.maximum(pre[:,1], ground[:,1])

    tiou = (e_i - b_i) / (e_u - b_u)

    for i in range(len(tiou)):
        if tiou[i] < 0.9:
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

    print cal_mAP('/Volumes/Transcend/mat/emotion6/baseline/svm_position.txt','/Volumes/Transcend/mat/emotion6/baseline/test_position.txt');
