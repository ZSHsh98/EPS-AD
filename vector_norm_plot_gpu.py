# from eagerpy import tile
# from bleach import clean
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import os 
import torch
import seaborn as sns
from sklearn import metrics
import torch.nn.functional as F
# revise the following arguments
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# t = 100
# 'CW']#[
attack_methods=['FGSM_L2']#['PGD','FGSM', 'CW', 'BIM',  'MIM', 'TIM', 'DI_MIM', 'FGSM_L2', 'BIM_L2', 'PGD_L2']
# attack_method = 'PGD'
dataset = 'imagenet' #'cifar' 'imagenet'
perb_image = True
isperb_image = 'perb_image' if perb_image else ''
epsilon = 0.01568 # 0.01568 |0.0313725
print('dataset:',dataset, 'epsilon:', epsilon)
for attack_method in attack_methods:
	print(f"======attack_method: {attack_method}")
	for t in [50]:
		with torch.no_grad():
			tile_name = f'scores_mmd_single_vector_norm_clean_adv_{attack_method}_{epsilon}_5_{t}{isperb_image}_naive'
			# tile_name = f'scores_clean_adv_mmd_APGD0.3_5_t_0-{t}'
			print(f"=========t: {t}")

			def plot_mi(clean, adv, path, name):

				mi_nat = clean.numpy()
				label_clean = 'Clean'

				mi_svhn = adv.numpy()
				label_adv = 'Adv'

				# fig = plt.figure()

				mi_nat = mi_nat[~np.isnan(mi_nat)]
				mi_svhn = mi_svhn[~np.isnan(mi_svhn)]

				# Draw the density plot
				sns.distplot(mi_nat, hist = True, kde = True,
							kde_kws = {'shade': True, 'linewidth': 1},
							label = label_clean)
				sns.distplot(mi_svhn, hist = True, kde = True,
							kde_kws = {'shade': True, 'linewidth': 1},
							label = label_adv)

				x = np.concatenate((mi_nat, mi_svhn), 0)
				y = np.zeros(x.shape[0])
				y[mi_nat.shape[0]:] = 1

				ap = metrics.roc_auc_score(y, x)
				fpr, tpr, thresholds = metrics.roc_curve(y, x)
				accs = {th: tpr[np.argwhere(fpr <= th).max()] for th in [0.01, 0.05, 0.1]}

				# Plot formatting
				# plt.legend()#(prop={'size': 20})
				# plt.xlabel('Information uncertainty')#, fontsize=20)
				# plt.ylabel('Density')#, fontsize=20)
				# plt.tight_layout()
				# plt.title(f'{name}-auroc:{int (ap*100000)/100000}')
				# plt.savefig(os.path.join(path, f'{label_clean}_vs_{label_adv}_{name}.pdf'), bbox_inches='tight')
				# plt.close()
				return "auroc: {:.4f}; ".format(ap) + "; ".join(["TPR: {:.4f} @ FPR={:.4f}".format(v, k) for k, v in accs.items()]) + "  {}-{}".format(len(mi_nat), len(mi_svhn))

			# # scores_f_fg [1000,500]
			# path = f'score_diffusion_t_{dataset}/scores_clean_norm1000_50samples.npy'
			# path_adv = f'score_diffusion_t_{dataset}/scores_adv_{attack_method}_0.01568_5_norm1000.npy'

			if dataset=='imagenet':
				path = f'score_diffusion_t_{dataset}/scores_cleansingle_vector_norm{t}{isperb_image}.npy'
				path_adv = f'score_diffusion_t_{dataset}/scores_adv_{attack_method}_{epsilon}_5single_vector_norm{t}{isperb_image}.npy'
			elif dataset=='cifar':
				# path = 'score_diffusion_t_cifar/scores_clean.npy'
				# path_adv = f'score_diffusion_t_{dataset}/scores_adv_{attack_method}_{epsilon}.npy'
				path = f'score_diffusion_t_cifar/scores_clean200{isperb_image}.npy'
				path_adv = f'score_diffusion_t_{dataset}/scores_adv_{attack_method}_{epsilon}_5200{isperb_image}.npy'

			log_dir = f'./score_diffusion_t_{dataset}/'
			os.makedirs(log_dir, exist_ok=True)
			x = torch.from_numpy(np.load(path)).cuda()
			x_adv = torch.from_numpy(np.load(path_adv)).cuda()
			
			# [200, 500,3,32,32]
			if dataset == 'cifar' or len(x.shape)!=4:
				x = x[:t,:,:,:,:].mean(0)#.half() 
				x_adv = x_adv[:t,:,:,:,:].mean(0)#.half()
			
			if dataset == 'imagenet':
				x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False).half()
				x_adv = F.interpolate(x_adv, size=(32, 32), mode='bilinear', align_corners=False).half()


			# path = f'score_diffusion_t_{dataset}/scores_clean.npy'
			# # path_adv = 'f'score_diffusion_t_{dataset}/scores_adv_{attack_method}_0.01568_5.npy''
			# path_adv = 'score_diffusion_t_cifar/scores_adv.npy'
			# log_dir = f'./score_diffusion_scalar_t_{dataset}/'
			# os.makedirs(log_dir, exist_ok=True)
			# x = torch.from_numpy(np.load(path))#.cuda()
			# x_adv = torch.from_numpy(np.load(path_adv))#.cuda()
			# x = x.view(*x.shape[:2],-1).norm(dim=-1).cuda()
			# x_adv = x_adv.view(*x_adv.shape[:2],-1).norm(dim=-1).cuda()

			# ############ calculate the refernce vetor by the dimention of samples via norm
			# # 1) calculate bar_x_norm [1000,1]
			# # bar_x_norm = x.view(x.shape[0], x.shape[1], -1).norm(dim=-1).mean(1, keepdims=True)
			# bar_x_norm = x #[1000, 500]
			# # 2) calculate mmd distance dt [500]
			# x_norm = x #[1000, 500]
			# x_adv_norm = x_adv #[1000, 500]
			# from cauculate_MMD import mmd
			# def dis_ref_clean2data(ref_clean, data):
			# 	print(f"The dimension of feature is {ref_clean.shape[0]}")
			# 	assert len(ref_clean.shape)==2 and len(data.shape)==2
			# 	dis_vector = torch.zeros_like(data[0],device=data.device)
			# 	for i in range(data.shape[1]):
			# 		dis_vector[i] = mmd(ref_clean.transpose(0,1),data[:,i].unsqueeze(0))
			# 	return dis_vector

			# dt_clean = dis_ref_clean2data(bar_x_norm[:t,:], x_norm[:t,:]).cpu()
			# dt_adv = dis_ref_clean2data(bar_x_norm[:t,:], x_adv_norm[:t,:]).cpu()

			dt_clean = x.view(x.shape[0],-1).norm(dim=-1).cpu()
			dt_adv = x_adv.view(x_adv.shape[0],-1).norm(dim=-1).cpu()

			print(plot_mi(dt_clean, dt_adv, log_dir, tile_name))
			# num_bins = 50
			# plt.hist(dt_clean, num_bins, facecolor='red', density=True, alpha=0.9)
			# plt.hist(dt_adv, num_bins,  facecolor='blue', density=True,alpha=0.9)
			# plt.title(f'{tile_name}')
			# plt.savefig(f'{log_dir}/{tile_name}.jpg')
			# plt.close()

# tile_name = f'scores_mmd_ref_mean_clean_adv_{attack_method}_0.01568_5_{t}_norm'

# def plot_mi(clean, adv, path, name):

# 	mi_nat = clean.numpy()
# 	label_clean = 'Clean'

# 	mi_svhn = adv.numpy()
# 	label_adv = 'Adv'

# 	# fig = plt.figure()

# 	mi_nat = mi_nat[~np.isnan(mi_nat)]
# 	mi_svhn = mi_svhn[~np.isnan(mi_svhn)]

# 	# Draw the density plot
# 	sns.distplot(mi_nat, hist = True, kde = True,
# 				 kde_kws = {'shade': True, 'linewidth': 1},
# 				 label = label_clean)
# 	sns.distplot(mi_svhn, hist = True, kde = True,
# 				 kde_kws = {'shade': True, 'linewidth': 1},
# 				 label = label_adv)

# 	x = np.concatenate((mi_nat, mi_svhn), 0)
# 	y = np.zeros(x.shape[0])
# 	y[mi_nat.shape[0]:] = 1

# 	ap = metrics.roc_auc_score(y, x)
# 	fpr, tpr, thresholds = metrics.roc_curve(y, x)
# 	accs = {th: tpr[np.argwhere(fpr <= th).max()] for th in [0.01, 0.05, 0.1]}

# 	# Plot formatting
# 	plt.legend()#(prop={'size': 20})
# 	plt.xlabel('Information uncertainty')#, fontsize=20)
# 	plt.ylabel('Density')#, fontsize=20)
# 	plt.tight_layout()
# 	plt.title(f'{name}-auroc:{int (ap*100000)/100000}')
# 	plt.savefig(os.path.join(path, f'{label_clean}_vs_{label_adv}_{name}.pdf'), bbox_inches='tight')
# 	plt.close()
# 	return "auroc: {:.4f}; ".format(ap) + "; ".join(["TPR: {:.4f} @ FPR={:.4f}".format(v, k) for k, v in accs.items()]) + "  {}-{}".format(len(mi_nat), len(mi_svhn))

# # scores_f_fg [1000,500]
# path = 'score_diffusion_t_cifar/scores_clean_norm.npy'
# path_adv = f'score_diffusion_t_cifar/scores_adv_{attack_method}_0.01568_5_norm.npy'
# log_dir = './score_diffusion_t_cifar/'
# os.makedirs(log_dir, exist_ok=True)
# x = torch.from_numpy(np.load(path)).cuda()
# x_adv = torch.from_numpy(np.load(path_adv)).cuda()

# ############ calculate the refernce vetor by the dimention of samples via norm
# # 1) calculate bar_x_norm [1000,1]
# # bar_x_norm = x.view(x.shape[0], x.shape[1], -1).norm(dim=-1).mean(1, keepdims=True)
# bar_x_norm = x #[1000, 500]
# # 2) calculate mmd distance dt [500]
# x_norm = x #[1000, 500]
# x_adv_norm = x_adv #[1000, 500]
# from cauculate_MMD import mmd
# def dis_ref_clean2data(ref_clean, data):
# 	print(f"The dimension of feature is {ref_clean.shape[0]}")
# 	assert len(ref_clean.shape)==2 and len(data.shape)==2
# 	dis_vector = torch.zeros_like(data[0],device=data.device)
# 	for i in range(data.shape[1]):
# 		dis_vector[i] = mmd(ref_clean.transpose(0,1),data[:,i].unsqueeze(0))
# 	return dis_vector



# dt_clean = dis_ref_clean2data(bar_x_norm[:t,:], x_norm[:t,:]).cpu()
# dt_adv = dis_ref_clean2data(bar_x_norm[:t,:], x_adv_norm[:t,:]).cpu()

# print(plot_mi(dt_clean, dt_adv, log_dir, tile_name))
# num_bins = 50
# plt.hist(dt_clean, num_bins, facecolor='red', density=True, alpha=0.9)
# plt.hist(dt_adv, num_bins,  facecolor='blue', density=True,alpha=0.9)
# plt.title(f'{tile_name}')
# plt.savefig(f'{log_dir}/{tile_name}.jpg')
# plt.close()