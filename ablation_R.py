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

attack_methods=['FGSM_L2'] #'PGD','FGSM', 'BIM',  'MIM', 'TIM', 'CW', 'DI_MIM', 'PGD_L2', 'FGSM_L2', 'BIM_L2', 'MM_Attack', 'AA_Attack']#['PGD','FGSM', 'CW', 'BIM',  'MIM', 'TIM', 'DI_MIM', 'FGSM_L2', 'BIM_L2', 'PGD_L2']
# attack_method = 'PGD'
dataset = 'cifar' #'cifar' 'imagenet'
perb_image = False
isperb_image = 'perb_image' if perb_image else ''
for num_sub in [100,200,300,400,500]:
	data_size = '' if num_sub==500 else str(num_sub)
	for epsilon in [0.01569]:  #[0.00392, 0.00784, 0.01176, 0.01569, 0.01961, 0.02353, 0.02745, 0.03137]:
		print('dataset:',dataset, 'epsilon:', epsilon)
		for attack_method in attack_methods:
			print(f"======attack_method: {attack_method}")
			for t in [5]: #1, 2, 5, 10, 20, 50, 100
				with torch.no_grad():
					tile_name = f'scores_mmd_single_vector_norm_clean_adv_{attack_method}_{epsilon}_5_{t}{isperb_image}_naive'
					nameROC = f'{attack_method}_{epsilon}'
					# tile_name = f'scores_clean_adv_mmd_APGD0.3_5_t_0-{t}'
					print(f"=========t: {t}")

					def plot_mi(clean, adv, path, name, nameROC):

						def plot_roc_curve(fper, tper,path, tile):
							plt.plot(fper, tper, color='red', label='ROC')
							plt.plot([0, 1], [0, 1], color='green', linestyle='--')
							plt.xlabel('False Positive Rate')
							plt.ylabel('True Positive Rate')
							plt.title(f'ROC_{tile}')
							plt.legend()
							plt.savefig(path)
							plt.close()
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
						# plot_roc_curve(fpr, tpr, f'{path}/{name}_ROC.jpg', nameROC)
						return "auroc: {:.4f}; ".format(ap) + "; ".join(["TPR: {:.4f} @ FPR={:.4f}".format(v, k) for k, v in accs.items()]) + "  {}-{}".format(len(mi_nat), len(mi_svhn))

					# if dataset== 'cifar':
					# 	path = f'/mnt/cephfs/ec/home/zhangshuhai/score_diffusion_t_cifar/scores_clean200.npy'
					# 	path_adv =  f'score_norm_{dataset}_t_ablation/scores_adv_{attack_method}_{epsilon}_5_norm{t}{isperb_image}.npy'
					# elif  dataset== 'imagenet':
					# 	path = f'score_norm_{dataset}_t_ablation/scores_clean_norm{t}{isperb_image}.npy'
					# 	path_adv =  f'score_norm_{dataset}_t_ablation/scores_adv_{attack_method}_{epsilon}_5_norm{t}{isperb_image}.npy'
					if dataset == 'cifar':
						# path = f'score_norm_cifar_t_ablation/scores_clean_norm{int(t*1000)/1000}{isperb_image}.npy'
						# path_adv =  f'score_norm_cifar_t_ablation/scores_adv_{attack_method}_{epsilon}_5_norm{int(t*1000)/1000}{isperb_image}.npy'
						path = f'size_ablation_{dataset}_R/scores_clean_norm{t}.0{isperb_image}{data_size}.npy'
						path_adv = f'size_ablation_{dataset}_R/scores_adv_{attack_method}_{epsilon}_5_norm{t}.0{isperb_image}{data_size}.npy'
					elif dataset == 'imagenet':
						# path = f'score_norm_imagenet_t_ablation/scores_clean_norm{int(t*1000)/1000}{isperb_image}.npy'
						# path_adv =  f'score_norm_imagenet_t_ablation/scores_adv_{attack_method}_{epsilon}_5_norm{int(t*1000)/1000}{isperb_image}.npy'
						path = f'size_ablation_resnet50_R/scores_clean_norm{t}.0{isperb_image}{data_size}.npy'
						path_adv = f'size_ablation_resnet50_R/scores_adv_{attack_method}_{epsilon}_5_norm{t}.0{isperb_image}{data_size}.npy'
					
					log_dir = f'./score_norm_{dataset}_t_ablation/'
					os.makedirs(log_dir, exist_ok=True)
					x = torch.from_numpy(np.load(path))#.cuda()
					x_adv = torch.from_numpy(np.load(path_adv))#.cuda()

					# if dataset== 'cifar':					
					# 	x = x[t:,:,:,:].reshape(500,-1).norm(dim=-1).reshape(-1)
					# 	x_adv = x_adv[-1]
					# elif dataset== 'imagenet':
					# 	x = x.reshape(-1)
					# 	x_adv = x_adv.reshape(-1)
					
					print(plot_mi(x, x_adv, log_dir, tile_name, nameROC))
                