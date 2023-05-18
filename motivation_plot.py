from turtle import color
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os 
import torch

dataset = 'imagenet' # 'cifar','imagenet'
print('dataset:',dataset)
attack_methods =  ['FGSM', 'PGD','FGSM_L2']
# scores_f_fg [1000,500,3,32,32]
perb_image = False
isperb_image = 'perb_image' if perb_image else ''
for attack_method in attack_methods:
	if dataset == 'imagenet':
		path = f'score_diffusion_t_{dataset}_motivation/scores_clean_norm50{isperb_image}.npy'
		path_adv = f'score_diffusion_t_{dataset}_motivation/scores_adv_{attack_method}_0.01568_5_norm50{isperb_image}.npy'
		log_dir = f'score_diffusion_t_{dataset}_motivation/'
	elif dataset == 'cifar':
		path = f'score_diffusion_t_{dataset}_motivation/scores_clean_norm50{isperb_image}.npy'
		path_adv = f'score_diffusion_t_{dataset}_motivation/scores_adv_{attack_method}_0.01568_5_norm50{isperb_image}.npy'
		log_dir = f'score_diffusion_t_{dataset}_motivation/'
		
	os.makedirs(log_dir, exist_ok=True)
	x = torch.from_numpy(np.load(path))
	x_adv = torch.from_numpy(np.load(path_adv))

	a = 0
	b = 50
	tile_name = f'scores_clean_adv_{attack_method}0.15_5_t_{a}-{b}_{isperb_image}'

	diff_x_adv_x_sum = 0
	num = x_adv.shape[1]
	# for i in range(20):
	# 	clean = torch.log(x[a:b,i])
	# 	adv =  torch.log(x_adv[a:b,i])
	# 	# diff_x_adv_x_sum += sum(adv > clean)

	# 	plt.plot(clean, color='red')
	# 	plt.plot(adv, color='blue')
	# 	plt.xlabel('Timestep/1000')
	# 	plt.ylabel('Score norm/log')
	# 	# plt.title(f'scores_t_{a}-{b}')
	# 	plt.savefig(f'{log_dir}/{tile_name}.jpg') # 
	# plt.close()
	num = 500
	x = x[:,0:num]
	x_adv = x_adv[:,0:num]
	clean_adv_max = torch.log(torch.cat([x,x_adv],dim=-1)).max(dim=-1)[0]
	clean = torch.log(x)/clean_adv_max.unsqueeze(dim=1)
	x_adv = torch.log(x_adv)/clean_adv_max.unsqueeze(dim=1)
	# clean_adv_max = torch.cat([x,x_adv],dim=-1).max(dim=-1)[0]
	# clean = x/clean_adv_max.unsqueeze(dim=1)
	# x_adv =x_adv/clean_adv_max.unsqueeze(dim=1)
	# x_range = np.linspace(min, max , 1)
	for i in range(0,len(clean),3):
		for j in range(len(clean[0])):
			plt.scatter(i, clean[i][j], color='aquamarine',marker='_')
			plt.scatter(i+1, x_adv[i][j], color='dodgerblue',marker='_')
	plt.xlabel('Timestep/1000')
	plt.ylabel('Score norm/log')
	plt.savefig(f'{log_dir}/{tile_name}.jpg') # 
	print(f'{log_dir}/{tile_name}.jpg')
	plt.close()
