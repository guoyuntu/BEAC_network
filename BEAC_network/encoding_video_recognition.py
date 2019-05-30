'''
function [features]= encoding_video_recogniton(centers, featurescell,opt)
%centers = twitter_C; 
% centers: num_center*num_dim(4096);
% featurescell: featurescell{1}: number_instances*num_dim(4096, by default);

m = size(centers,1);
%feature = zeros(1,50);
% k=1;
% for wi = 1:m 
%     if sum(centers(wi,:).^2)~=0
%         centers(wi,:)=centers(wi,:)/sqrt(sum(centers(wi,:).^2));
%     end
% end       


% l2-sum-1:
coef =4000;
l2 =@(x) x./(repmat(sqrt(sum(x.*x,2)./coef),1,size(x,2))+eps);

videoNum = size(featurescell,1);
for i = 1:videoNum
     videofeat=featurescell(i,:,:);
     bow = bowcal(videofeat,centers,opt);
     features(i,:)=bow/size(videofeat,1);
;
end

end

function bow = bowcal(fea,centers,opt)
% compute bow, given fea, centers, and m
bow = zeros(1,size(centers,1));
simi = fea*centers';
m=size(fea,1);
% Calculate similarity with all vk.
 for wi = 1:m 
     if sum(fea(wi,:).^2)~=0
         fea(wi,:)=fea(wi,:)/sqrt(sum(fea(wi,:).^2));
     end
 end       
for wi=1:m
          weight=linspace(1,0,opt.bin);
         %weight=ones(1,opt.bin);
            for d=1:length(weight)
              [Y I]=max(simi(wi,:));
                bow(1,I)=bow(1,I)+Y*weight(d);
                simi(wi,I)=0;
            end
   

    
end
end
'''
import numpy as np
from numpy.matlib import repmat
import read_mat
import h5py

def encoding_video_recognition(centers, featurescell, opt):
	feature = np.zeros((np.shape(featurescell)[0], 2000))
	videoNum = np.shape(featurescell)[0]
	for i in range(videoNum):
		print(i)
		videofeat = featurescell[i]
		bow = bowcal(videofeat,centers,opt)
		feature[i] = bow / np.shape(videofeat)[0]
	return feature

def bowcal(fea, centers, opt):
	bow = np.zeros(np.shape(centers)[1])
	simi = np.dot(fea, centers)
	m = np.shape(fea)[0]
	for wi in range(m):
		if (fea[wi] * fea[wi]).sum() != 0:
			fea[wi] = fea[wi] / np.sqrt((fea[wi] * fea[wi]).sum())
	for wi in range(m):
		weight = np.linspace(0, 1, opt)
		for d in range(len(weight)):
			y = np.max(simi[wi])
			l = np.where(simi[wi] == y)[0][0]
			bow[l] = bow[l] + y * weight[d]
			simi[wi][l] = 0
	return bow
'''
flickr = h5py.File('/home/g_jiarui/video_spacial_tu/code/mat/flickr_cluster_centers_2000.mat')['centers'][:].T
coef = 4000

l2 = lambda x: x / (repmat(np.sqrt((x * x).sum(axis = 1) / coef), np.shape(x)[1], 1).T + np.finfo(np.float32).eps)
norm_flickr=l2(flickr)
features = read_mat.read_mat('/home/g_jiarui/merge_fp/data/train_data_100.mat')['train_data']
opt = 190
#c_t = encoding_video_recognition(norm_flickr, features, opt)
a = bowcal(features[0], norm_flickr, opt)
print(a)
'''