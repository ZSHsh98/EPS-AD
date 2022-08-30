
import torch
import numpy as np


def calc_k(S, percentge):
	'''确定k值：前k个奇异值的平方和占比 >=percentage, 求满足此条件的最小k值
	:param percentage, 奇异值平方和的占比的阈值
	:return 满足阈值percentage的最小k值
	'''
	k = 0
	#用户数据矩阵的奇异值序列的平方和
	total = sum(torch.square(S))
	svss = 0 #奇异值平方和 singular values square sum
	for i in range(S.shape[0]):
		svss += torch.square(S[i])
		if (svss/total) >= percentge:
			k = i+1
			break
	return k

def PCA_svd(x, x_adv, center=True):
	x_mean = x.mean(dim=-1, keepdim=True)
	# x_var = x.var(dim=1, keepdim=True)
	# x_center = (x - x_mean)/x_var
	# x_adv_ceter = (x_adv - x_mean)/x_var
	x_center =  torch.matmul((x - x_mean).T, x - x_mean)/x.shape[0]
	# x_adv_ceter = torch.matmul((x_adv - x_mean).T, x_adv - x_mean)/x_adv.shape[0]

	_, s, v = torch.svd(x_center)
	k = calc_k(s, percentge=0.9999)
	print(f'the dimension: {x.shape[1]} -->{k}')

	# SD = torch.eye(k) * s[:k]

	pca_x = x.mm(v[:,:k])
	pca_x_adv = x_adv.mm(v[:,:k])
	
	return pca_x, pca_x_adv



