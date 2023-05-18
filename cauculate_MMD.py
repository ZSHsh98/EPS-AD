import numpy as np
import torch

def guassian_kernel(source, target, kernel_mul, kernel_num=5, fix_sigma=None):
	"""计算Gram核矩阵
	source: sample_size_1 * feature_size 的数据
	target: sample_size_2 * feature_size 的数据
	kernel_mul: 这个概念不太清楚,感觉也是为了计算每个核的bandwith
	kernel_num: 表示的是多核的数量
	fix_sigma: 表示是否使用固定的标准差
		return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
						矩阵，表达形式:
						[   K_ss K_st
							K_ts K_tt ]
	"""
	n_samples = int(source.size()[0])+int(target.size()[0])
	total = torch.cat([source, target], dim=0) # 合并在一起

	total0 = total.unsqueeze(0).expand(int(total.size(0)), \
									   int(total.size(0)), \
									   int(total.size(1)))
	total1 = total.unsqueeze(1).expand(int(total.size(0)), \
									   int(total.size(0)), \
									   int(total.size(1)))
	value_factor = 15
	L2_distance = (((total0-total1)/value_factor)**2).sum(2) # 计算高斯核中的|x-y|
	# L2_distance = ((total0-total1)**2).sum(2) # 计算高斯核中的|x-y|
	assert not torch.isinf(L2_distance).any(), 'tune the value_factor larger'

	# 计算多核中每个核的bandwidth
	if fix_sigma:
		bandwidth = fix_sigma
	else:
		# bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth = torch.sum(L2_distance.data / ((n_samples**2-n_samples)/(value_factor)**2) /(value_factor)**2)
		assert not torch.isinf(bandwidth).any()
	bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
	scale_factor = 0
	bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
	# bandwidth_list = [torch.clamp(bandwidth * (kernel_mul**scale_factor) *(kernel_mul**(i-scale_factor)),max=1.0e38) for i in range(kernel_num)]

	# 高斯核的公式，exp(-|x-y|/bandwith)
	kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
				  bandwidth_temp in bandwidth_list]

	return sum(kernel_val)/kernel_num # 将多个核合并在一起

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
	n = int(source.size()[0])
	m = int(target.size()[0])

	kernels = guassian_kernel(source, target,
							  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
	XX = kernels[:n, :n] 
	YY = kernels[n:, n:]
	XY = kernels[:n, n:]
	YX = kernels[n:, :n]

	XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # K_ss矩阵，Source<->Source
	XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # K_st矩阵，Source<->Target

	YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) # K_ts矩阵,Target<->Source
	YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # K_tt矩阵,Target<->Target
		
	loss = (XX + XY).sum() + (YX + YY).sum()
	# loss = XY.sum()
	return loss

# def L2_distance_get(A, B, value_factor = 1):
# 	num_A = A.shape[0]
# 	num_B = B.shape[0]
# 	L2_distance = torch.zeros((num_B, num_A),device=A.device)
# 	for i in range(num_B):
# 		L2_distance[i] = ((((B[i]).unsqueeze(0) - A)/value_factor)**2).sum(dim=-1)
# 		assert (L2_distance[i]).shape[0]==num_A
# 	return L2_distance

def L2_distance_get(x, y, value_factor = 1):
	"""compute the paired distance between x and y."""
	x_norm = ((x/value_factor) ** 2).sum(1).view(-1, 1)
	y_norm = ((y/value_factor) ** 2).sum(1).view(1, -1)
	Pdist = x_norm + y_norm - 2.0 * torch.mm(x/value_factor, torch.transpose(y/value_factor, 0, 1))
	Pdist[Pdist<0]=0
	return Pdist

def mmd_guassian_bigtensor(L2_distance_xx, L2_distance_yx, value_factor = 15, kernel_num=5, kernel_mul=2.0,  fix_sigma=None, clean_flag=False):


	x_num = L2_distance_xx.shape[0]
	y_num = L2_distance_yx.shape[0]
	assert y_num==1 and L2_distance_yx.shape[1]==x_num
	
	# 计算多核中每个核的bandwidth
	if fix_sigma:
		bandwidth = fix_sigma
	else:
		# bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth = (torch.sum(L2_distance_xx/x_num) + torch.sum(L2_distance_yx/x_num))/(x_num-1 + L2_distance_yx.shape[1])
		# bandwidth = torch.sum(L2_distance.data / ((n_samples**2-n_samples)/(value_factor)**2) /(value_factor)**2)
		assert not torch.isinf(bandwidth).any()
	# bandwidth /= kernel_mul ** (kernel_num // 2)
	bandwidth = bandwidth / kernel_mul ** (kernel_num // 2)
	# print("bandwidth:",bandwidth)
	bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
	# 高斯核的公式，exp(-|x-y|/bandwith)
	XX = sum([torch.exp(-L2_distance_xx / bandwidth_temp) for \
				  bandwidth_temp in bandwidth_list])/kernel_num
	YX = sum( [torch.exp(-L2_distance_yx / bandwidth_temp) for \
				bandwidth_temp in bandwidth_list])/kernel_num

	L2_distance = (XX/x_num).sum()/x_num - 2*YX.sum()/x_num
	return L2_distance

def mmd_guassian_bigtensor2(L2_distance_xx, L2_distance_yx, value_factor = 15, kernel_num=5, kernel_mul=2.0,  fix_sigma=None, clean_flag=False):


	x_num = L2_distance_xx.shape[0]
	y_num = L2_distance_yx.shape[0]
	assert y_num==1 and L2_distance_yx.shape[1]==x_num
	
	bandwidth = fix_sigma
	bandwidth /= kernel_mul ** (kernel_num // 2)
	# print("bandwidth:",bandwidth)
	bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
	# 高斯核的公式，exp(-|x-y|/bandwith)
	YX = sum( [torch.exp(-L2_distance_yx / bandwidth_temp) for \
				bandwidth_temp in bandwidth_list])/kernel_num

	L2_distance = - 2*YX.sum()/x_num
	return L2_distance

def mmd_guassian_kernel_batch(ref_data, test_data, kernel_num=5, kernel_mul=2.0, fix_sigma=None, clean_flag=False):

	num_ref = ref_data.shape[0]
	# if clean_flag:
	# 	num_ref = ref_data.shape[0]-1
	num_test = test_data.shape[0]
	# ref_data_exp = ref_data.unsqueeze(0)
	# test_data_exp = test_data.unsqueeze(1)
	value_factor = 1
	# L2_distance = (((ref_data_exp-test_data_exp)/value_factor)**2).sum(2) # 计算高斯核中的|x-y|
	L2_distance = L2_distance_get(test_data, ref_data, value_factor)

	assert not torch.isinf(L2_distance).any(), 'tune the value_factor larger'

	# 计算多核中每个核的bandwidth
	if fix_sigma:
		bandwidth = fix_sigma
	else:
		# bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth = torch.sum(L2_distance.data / ((num_ref)/(value_factor)**2) /(value_factor)**2,dim=-1)
		assert not torch.isinf(bandwidth).any()
	bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
	assert bandwidth.shape[0]==num_test
	bandwidth_alllist = []
	scale_factor = 1
	for id_test in range(num_test):
		# bandwidth_list = [torch.clamp(bandwidth[id_test] * (kernel_mul**scale_factor) *(kernel_mul**(i-scale_factor)),max=1.0e38) for i in range(kernel_num)]
		bandwidth_list = [bandwidth[id_test] * (kernel_mul**scale_factor) *(kernel_mul**(i-scale_factor)) for i in range(kernel_num)]
		bandwidth_alllist.append(bandwidth_list)
	bandwidth_alltensor = torch.tensor(bandwidth_alllist,device=test_data.device)
	assert bandwidth_alltensor.shape[1]==kernel_num
	L2_distance_all = 0
	for k in range(kernel_num):
		L2_distance_all += (torch.exp(-L2_distance / bandwidth_alltensor[:,k].unsqueeze(1)))
		# L2_distance_all += -L2_distance / bandwidth_alltensor[:,k].unsqueeze(1)
	assert L2_distance_all.shape[0]==num_test

	# return L2_distance_all.sum(dim=-1)/(-kernel_num*num_ref)
	return L2_distance_all.sum(dim=-1)/(-num_ref)

	# 高斯核的公式，exp(-|x-y|/bandwith)
	# kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
	# 			  bandwidth_temp in bandwidth_list]

	# return sum(kernel_val)/kernel_num # 将多个核合并在一起
	
def mmd_guassian_kernel_single(L2_distance=0.0,value_factor = 15, kernel_num=5, kernel_mul=2.0,  fix_sigma=None, clean_flag=False):

	num_ref = L2_distance.shape[1]
	num_test = L2_distance.shape[0]
	# ref_data_exp = ref_data.unsqueeze(0)
	# test_data_exp = test_data.unsqueeze(1)
	
	# L2_distance = (((ref_data_exp-test_data_exp)/value_factor)**2).sum(2) # 计算高斯核中的|x-y|

	assert not torch.isinf(L2_distance).any(), 'tune the value_factor larger'

	# 计算多核中每个核的bandwidth
	if fix_sigma is not None:
		bandwidth0 = fix_sigma.expand(int(num_test))
	else:
		# bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth0 = torch.sum(L2_distance.data / ((num_ref)/(value_factor)**2) /(value_factor)**2,dim=-1)
		assert not torch.isinf(bandwidth).any()
	bandwidth = bandwidth0/ kernel_mul ** (kernel_num // 2)
	assert bandwidth.shape[0]==num_test
	bandwidth_alllist = []
	scale_factor = 1
	for id_test in range(num_test):
		# bandwidth_list = [torch.clamp(bandwidth[id_test] * (kernel_mul**scale_factor) *(kernel_mul**(i-scale_factor)),max=1.0e38) for i in range(kernel_num)]
		bandwidth_list = [bandwidth[id_test] * (kernel_mul**scale_factor) *(kernel_mul**(i-scale_factor)) for i in range(kernel_num)]
		bandwidth_alllist.append(bandwidth_list)
	bandwidth_alltensor = torch.tensor(bandwidth_alllist,device=L2_distance.device)
	assert bandwidth_alltensor.shape[1]==kernel_num
	L2_distance_all = 0
	for k in range(kernel_num):
		L2_distance_all += (torch.exp(-L2_distance / bandwidth_alltensor[:,k].unsqueeze(1)))
	assert L2_distance_all.shape[0]==num_test

	# return L2_distance_all.sum(dim=-1)/(-kernel_num*num_ref)
	return L2_distance_all.sum(dim=-1)/(-num_ref)

# if __name__ == "__main__":
#     # 样本数量可以不同，特征数目必须相同

#     # 100和90是样本数量，50是特征数目
#     data_1 = torch.tensor(np.random.normal(loc=0,scale=10,size=(100,50)))
#     data_2 = torch.tensor(np.random.normal(loc=10,scale=10,size=(90,50)))
#     print("MMD Loss:",mmd(data_1,data_2))

#     data_1 = torch.tensor(np.random.normal(loc=0,scale=10,size=(100,50)))
#     data_2 = torch.tensor(np.random.normal(loc=0,scale=9,size=(80,50)))

#     print("MMD Loss:",mmd(data_1,data_2))

# MMD Loss: tensor(1.0866, dtype=torch.float64)
# MMD Loss: tensor(0.0852, dtype=torch.float64)