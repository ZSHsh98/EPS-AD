# from eagerpy import tile
# from bleach import clean
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import os 
import torch
import seaborn as sns
from sklearn import metrics

################### (method 2)

# os.environ["CUDA_VISIBLE_DEVICES"] = '7'
# t = 100
attack_methods=['PGD','FGSM', 'BIM',  'MIM', 'TIM', 'CW', 'DI_MIM', 'PGD_L2', 'FGSM_L2', 'BIM_L2', 'MM_Attack','AA_Attack'] #['PGD','FGSM', 'CW', 'BIM',  'MIM', 'TIM', 'DI_MIM', 'FGSM_L2', 'BIM_L2', 'PGD_L2']
# attack_method = 'PGD'
dataset = 'imagenet' # 'cifar',  'imagenet'
perb_image = False
isperb_image = 'perb_image' if perb_image else ''
stand_flag = True
isstand = '_stand' if stand_flag else ''
# epsilon = [0.00392, 0.00784, 0.01176, 0.01569, 0.01961, 0.02353, 0.02745, 0.03137] # 0.01568 |0.0313725|0.00392
# print('dataset:',dataset, 'epsilon:', epsilon)
# from PCASVD import PCA_svd
for epsilon in [0.00392, 0.00784, 0.01176, 0.01569, 0.01961, 0.02353, 0.02745, 0.03137]:
	print('dataset:',dataset, 'epsilon:', epsilon)
	for attack_method in attack_methods:
		print(f"======attack_method: {attack_method}")
		for t in [0.001]: #, 0.005, 0.01, 0.05, 0.1, 0.5, 1]:
			with torch.no_grad():
				tile_name = f'scores_mmd_single_vector_norm_clean_adv_{attack_method}_{epsilon}_5_{t}{isperb_image}'
				# tile_name = f'scores_clean_adv_APGD0.3_5_t_0-{t}'
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
					plt.legend()#(prop={'size': 20})
					plt.xlabel('Information uncertainty')#, fontsize=20)
					plt.ylabel('Density')#, fontsize=20)
					plt.tight_layout()
					plt.title(f'{name}-auroc:{int (ap*100000)/100000}')
					plt.savefig(os.path.join(path, f'{label_clean}_vs_{label_adv}_{name}.pdf'), bbox_inches='tight')
					plt.close()
					return "auroc: {:.4f}; ".format(ap) + "; ".join(["TPR: {:.4f} @ FPR={:.4f}".format(v, k) for k, v in accs.items()]) + "  {}-{}".format(len(mi_nat), len(mi_svhn))

				# scores_f_fg [500,3,32,32]
				if dataset=='imagenet':
					pass
					# path = f'score_diffusion_t_{dataset}/scores_cleansingle_vector_norm{t}{isperb_image}.npy'
					# path_adv = f'score_diffusion_t_{dataset}{isstand}/scores_adv_{attack_method}_{epsilon}_5single_vector_norm{t}{isperb_image}.npy'
					path = f'score_norm_cifar_imagenet_baseline/scores_clean_norm{int(t*1000)/1000}{isperb_image}.npy'
					path_adv = f'score_norm_cifar_imagenet_baseline/scores_adv_{attack_method}_{epsilon}_5_norm{int(t*1000)/1000}{isperb_image}.npy'
				elif dataset=='cifar':
					# path = 'score_diffusion_t_cifar/scores_clean.npy'
					# path_adv = f'score_diffusion_t_{dataset}/scores_adv_{attack_method}_{epsilon}.npy'
					path = f'score_norm_cifar_imagenet_baseline/scores_clean_norm{int(t*1000)/1000}{isperb_image}.npy'
					path_adv = f'score_norm_cifar_imagenet_baseline/scores_adv_{attack_method}_{epsilon}_5_norm{int(t*1000)/1000}{isperb_image}.npy'
				# path_adv = 'score_diffusion_t_cifar/scores_adv.npy'
				log_dir = f'score_norm_cifar_imagenet_baseline/'
				os.makedirs(log_dir, exist_ok=True)
				x = torch.from_numpy(np.load(path))#.cuda()
				x_adv = torch.from_numpy(np.load(path_adv))#.cuda()
						
				############ calculate the mean by the dimention of time 
				# 1) calculate bar_x_sigle_sample [1, 500, 3, 32, 32]
				if dataset == 'cifar':
					x = x#[:t,:,:,:,:].mean(0)#.half() 
					x_adv = x_adv#[:t,:,:,:,:].mean(0)#.half()
				
				if dataset == 'imagenet':
					pass
					# x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)#.half()
					# x_adv = F.interpolate(x_adv, size=(32, 32), mode='bilinear', align_corners=False)#.half()


				x = x.view(x.shape[0],-1)
				x_adv = x_adv.view(x_adv.shape[0],-1)

				dt_clean = x
				dt_adv = x_adv

				print(plot_mi(dt_clean, dt_adv, log_dir, tile_name))
				num_bins = 100
				plt.hist(dt_clean, num_bins, facecolor='red', density=True, alpha=0.9)
				plt.hist(dt_adv, num_bins,  facecolor='blue', density=True,alpha=0.9)
				plt.title(f'{tile_name}')
				plt.savefig(f'{log_dir}/{tile_name}.jpg')
				plt.close()