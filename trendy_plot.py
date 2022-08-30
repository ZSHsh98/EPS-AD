import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os 
import torch
'''path = 'image/scores.npy'
log_dir = './image'
os.makedirs(log_dir, exist_ok=True)
y = np.load(path)
plt.plot(y[:,2])
plt.savefig(f'{log_dir}/score_norm0.jpg')'''


# # scores_f_fg [1000,3,500]
# path = './image/purify_scores_f_fg/purify_scores_f_fg_clean.npy'
# path_adv = './image/purify_scores_f_fg/purify_scores_f_fg_adv.npy'
# log_dir = './image/purify_scores_f_fg'
# os.makedirs(log_dir, exist_ok=True)

# x = np.load(path)
# x_adv = np.load(path_adv)

# for i in range(500):
# 	clean = x[:,0,i]
# 	adv = x_adv[:,0,i]


# 	plt.plot(clean[:15], color='red')
# 	plt.plot(adv[:15], color='blue')
# 	plt.title('scores_clean_adv_0_15')
# 	plt.savefig(f'{log_dir}/scores_clean_adv_0_15.jpg')

# #socres(x,t) * beta(t)
# t = 1000 - 1
# beta_min = 0.1
# beta_max = 20
# N = 1000
# discrete_betas = np.linspace(beta_min/N , beta_max/N, N)
# alphas = 1. - discrete_betas
# alphas_cumprod = np.cumprod(alphas)
# sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
# ## reverse process: socres(x,t), t: 1000-->1, variance from big to small.
# betas = sqrt_alphas_cumprod[:t]
# for i in range(500):
#     clean = x[0:t,0,i] * betas
#     adv = x_adv[0:t,0,i]* betas

#     plt.plot(clean, color='red')
#     plt.plot(adv, color='blue')
#     plt.title(f'scores_beta_iter_t_0-{t}')
#     plt.savefig(f'{log_dir}/scores_beta_iter_t_0-{t}.jpg')
# plt.close()



'''# t_purify_scores_f_fg [20,100,3,500]
path = './image/t_purify_scores_f_fg/t_purify_scores_f_fg_clean.npy'
path_adv = './image/t_purify_scores_f_fg/t_purify_scores_f_fg_adv.npy'
log_dir = './image/t_purify_scores_f_fg'
os.makedirs(log_dir, exist_ok=True)

x = np.load(path)
x_adv = np.load(path_adv)

a = 0
b = 15
for iter in range(1, x.shape[0],3):
	for i in range(500):
		clean = x[iter,:,0,i]
		adv = x_adv[iter,:,0,i]

		plt.plot(clean[a:b], color='red')
		plt.plot(adv[a:b], color='blue')
		plt.title(f'f_clean_adv_iter_{iter}_form_{a}_{b}')
		plt.savefig(f'{log_dir}/f_clean_adv_iter_{iter}_form_{a}_{b}.jpg')
	plt.close()'''



# #differential t_purify_scores_f_fg [20,100,3,500]
# path = './image/t_purify_scores_f_fg/t_purify_scores_f_fg_clean.npy'
# path_adv = './image/t_purify_scores_f_fg/t_purify_scores_f_fg_adv.npy'
# log_dir = './image/differential t_purify_scores_f_fg'
# os.makedirs(log_dir, exist_ok=True)

# x = np.load(path)
# x_adv = np.load(path_adv)

# a = 0
# b = 30
# for iter in range(0, x.shape[0]-1,3):
# 	for i in range(500):
# 		clean = x[iter+1,:,0,i] - x[iter,:,0,i]
# 		adv = x_adv[iter+1,:,0,i] - x_adv[iter,:,0,i]

# 		plt.plot(clean[a:b], color='red')
# 		plt.plot(adv[a:b], color='blue')
# 		plt.title(f'scores_clean_adv_iter_{iter}_form_{a}_{b}')
# 		plt.savefig(f'{log_dir}/scores_clean_adv_iter_{iter}_form_{a}_{b}.jpg')
# 	plt.close()

### iter_diff
# for i in range(500):
#     clean = x[1:20,0,1,i] - x[0:19,0,1,i]
#     adv = x_adv[1:20,0,1,i] - x_adv[0:19,0,1,i]

#     plt.plot(clean, color='red')
#     plt.plot(adv, color='blue')
#     plt.title('f_iter_diff_0_19')
#     plt.savefig(f'{log_dir}/f_iter_diff_0_19.jpg')
# plt.close()

### iter_nodiff
# for i in range(500):
#     clean = x[0:20,0,1,i]
#     adv = x_adv[0:20,0,1,i]

#     plt.plot(clean, color='red')
#     plt.plot(adv, color='blue')
#     plt.title('f_iter_nodiff_0_19')
#     plt.savefig(f'{log_dir}/f_iter_nodiff_0_19.jpg')
# plt.close()

# # hist plot
# clean = x[1,0,1,:]- x[0,0,1,:]
# adv = x_adv[1,0,1,:]- x_adv[0,0,1,:]
# num_bins = 50
# plt.hist(clean, num_bins, facecolor='red', density=True, alpha=0.9)
# plt.hist(adv, num_bins,  facecolor='blue', density=True,alpha=0.9)
# plt.title('f_diff_clean_adv_t_1-0')
# plt.savefig(f'{log_dir}/f_diff_clean_adv_t_1-0.jpg')
# plt.close()

# #diff_iter {t=1 - t=0} in [0,100]
# t = 20
# for i in range(500):
#     clean = x[1,0:t,0,i] -x[0,0:t,0,i] 
#     adv = x_adv[1,0:t,0,i] - x_adv[0,0:t,0,i]

#     plt.plot(clean, color='red')
#     plt.plot(adv, color='blue')
#     plt.title(f'scores_diff_iter_1-0_t_0-{t}')
#     plt.savefig(f'{log_dir}/scores_diff_iter_1-0_t_0-{t}.jpg')
# plt.close()

# # socres(x,t) * beta(t)
# t = 100
# beta_min = 0.1
# beta_max = 20
# N = 1000
# discrete_betas = np.linspace(beta_min , beta_max, N)
# ## reverse process: socres(x,t), t: 1000-->1, variance from big to small.
# betas = discrete_betas[:t]
# for i in range(500):
#     clean = x[0,0:t,0,i] * betas
#     adv = x_adv[0,0:t,0,i]* betas

#     plt.plot(clean, color='red')
#     plt.plot(adv, color='blue')
#     plt.title(f'scores_beta_iter_t_0-{t}')
#     plt.savefig(f'{log_dir}/scores_beta_iter_t_0-{t}.jpg')
# plt.close()

# scores_f_fg [1000,3,500]
# path = 'images_other_setting/CIFAR10/scores_f_fg_clean.npy'
# path_adv = './images_other_setting/CIFAR10/scores_f_fg_adv.npy'
# log_dir = './images_other_setting/CIFAR10'
# os.makedirs(log_dir, exist_ok=True)
# x = np.load(path)
# x_adv = np.load(path_adv)
##socres(x,t) * beta(t)
# t = 1000
# beta_min = 0.1
# beta_max = 20
# N = 1000
# discrete_betas = np.linspace(beta_min/N , beta_max/N, N)
# alphas = 1. - discrete_betas
# alphas_cumprod = np.cumprod(alphas)
# sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
# ## reverse process: socres(x,t), t: 1000-->1, variance from big to small.
# betas = sqrt_alphas_cumprod[::-1][:t]
# plt.plot(betas, color='green')
# plt.savefig(f'{log_dir}/sqrt_alphas_cumprod_t_0-{t}.jpg')
# plt.close()
# for i in range(500):
#     clean = x[0:t,0,i] * betas
#     adv = x_adv[0:t,0,i]* betas

#     plt.plot(clean, color='red')
#     plt.plot(adv, color='blue')
#     plt.title(f'scores_beta_iter_t_0-{t}')
#     plt.savefig(f'{log_dir}/scores_beta_iter_t_0-{t}.jpg')
# plt.close()


'''# [1000,500]
path = 'score_diffusion_t_imagenet_debug/scores_clean_norm1000.npy'
path_adv = 'score_diffusion_t_imagenet_debug/scores_adv_PGD_0.01568_5_norm1000.npy'
log_dir = './score_diffusion_t_imagenet_debug/'
os.makedirs(log_dir, exist_ok=True)
x = torch.from_numpy(np.load(path))
x_adv = torch.from_numpy(np.load(path_adv))

# x = x.view(*x.shape[:2], -1).norm(dim=-1)
# x_adv = x_adv.view(*x_adv.shape[:2], -1).norm(dim=-1)

a = 0
b = 20
for i in range(100):
	clean = x[a:b,i] 
	adv = x_adv[a:b,i]

	# plt.plot(clean, color='red')
	plt.plot(adv, color='blue')
	plt.title(f'scores_t_{a}-{b}')
	plt.savefig(f'{log_dir}/scores_adv_PGD0.015686_5_t_{a}-{b}.jpg') # 
plt.close()'''

dataset = 'cifar' # 'cifar','imagenet'
print('dataset:',dataset)
attack_method = 'PGD'
# scores_f_fg [1000,500,3,32,32]
perb_image = True
isperb_image = 'perb_image' if perb_image else ''
if dataset == 'imagenet':
	path = f'score_diffusion_t_{dataset}/scores_clean_norm101.npy'
	# path_adv = f'score_diffusion_t_{dataset}/scores_adv_{attack_method}_0.01568_5.npy'
	path_adv = f'score_diffusion_t_{dataset}/scores_adv_{attack_method}_0.01568_5_norm101.npy'
	# path = f'score_diffusion_t_{dataset}/scores_cleansingle_vector_norm100{isperb_image}.npy'
	# path_adv = f'score_diffusion_t_{dataset}/scores_adv_{attack_method}_0.01568_5single_vector_norm100{isperb_image}.npy'
	log_dir = f'./score_diffusion_t_{dataset}/'
elif dataset == 'cifar':
	path = f'score_diffusion_t_cifar/scores_clean200{isperb_image}.npy'
	path_adv = f'score_diffusion_t_{dataset}/scores_adv_{attack_method}_0.01568_5200{isperb_image}.npy'
	log_dir = f'./score_diffusion_t_{dataset}/'
	
os.makedirs(log_dir, exist_ok=True)
x = torch.from_numpy(np.load(path))
x_adv = torch.from_numpy(np.load(path_adv))

if dataset == 'cifar':
	x = x.view(*x.shape[:2], -1).norm(dim=-1)
	x_adv = x_adv.view(*x_adv.shape[:2], -1).norm(dim=-1)

a = 0
b = 10
tile_name = f'scores_clean_adv_{attack_method}0.15_5_t_{a}-{b}_{isperb_image}'
# tile_name = f'scores_adv_{attack_method}0.3_5_t_0-{b}'
diff_x_adv_x_sum = 0
# for i in range(x_adv.shape[1]):
# 	clean = x[a:b,i] 
# 	adv = x_adv[a:b,i]
# 	# diff_x_adv_x_sum += sum(adv > clean)

# 	plt.plot(clean, color='red')
# 	plt.plot(adv, color='blue')
# 	plt.title(f'scores_t_{a}-{b}')
# 	plt.savefig(f'{log_dir}/{tile_name}.jpg') # 
# plt.close()

for t in range(50):
	print(sum(x[t,:]<x_adv[t,:]))