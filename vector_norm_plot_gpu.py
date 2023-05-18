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
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
# t = 100
# 'CW']#[
attack_methods=['FGSM_L2'] #'PGD','FGSM', 'BIM',  'MIM', 'TIM', 'CW', 'DI_MIM', 'PGD_L2', 'FGSM_L2', 'BIM_L2', 'MM_Attack', 'AA_Attack']
# attack_method = 'PGD'
dataset = 'cifar' #'cifar' 'imagenet'
perb_image = None
isperb_image = 'perb_image' if perb_image else ''
# epsilon = 0.01568 # 0.01568 |0.0313725
for num_sub in [500]:
	data_size = '' if num_sub==500 else str(num_sub)
	for epsilon in [0.01569]: #0.00392, 0.00784, 0.01176, 0.01569, 0.01961, 0.02353, 0.02745, 0.03137]:
		print('dataset:',dataset, 'epsilon:', epsilon)
		for attack_method in attack_methods:
			print(f"======attack_method: {attack_method}")
			# for tn in [0.1,0.2,0.3]+torch.linspace(0,20,41)[1:].tolist():
			# for tn in [20.0]:
			for tn in [10.0,15.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0]:
				tn = int(tn*10)
				# for t0 in [0.0,0.1,0.2,0.3]+torch.linspace(0,20,41)[1:].tolist():
				for t0 in [0.0]:
					t0 = int(t0*10)
					with torch.no_grad():
						tile_name = f'scores_mmd_single_vector_norm_clean_adv_{attack_method}_{epsilon}_5_{tn/10}{isperb_image}_naive'
						# tile_name = f'scores_clean_adv_mmd_APGD0.3_5_t_0-{t}'
						print(f"=========t0: {t0}, tn: {tn}")

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
							# path = f'score_diffusion_t_{dataset}/scores_cleansingle_vector_norm{t}{isperb_image}{data_size}.npy'
							# path_adv = f'score_diffusion_t_{dataset}/scores_adv_{attack_method}_{epsilon}_5single_vector_norm{t}{isperb_image}{data_size}.npy'
							# path = f'score_norm_{dataset}_t_ablation/scores_cleansingle_vector_norm{t}{isperb_image}{data_size}.npy'
							# path_adv = f'score_diffusion_t_{dataset}_stand/scores_adv_{attack_method}_{epsilon}_5single_vector_norm{t}{isperb_image}{data_size}.npy'
							# path = f'score_diffusion_t_{dataset}101/scores_cleansingle_vector_norm{t}{isperb_image}{data_size}.npy'
							# path_adv = f'score_diffusion_t_{dataset}101/scores_adv_{attack_method}_{epsilon}_5single_vector_norm{t}{isperb_image}{data_size}.npy'
							# path = f'size_ablation_resnet50_N/scores_cleansingle_vector_norm{t}{isperb_image}{data_size}.npy'
							# path_adv = f'size_ablation_resnet50_N/scores_adv_{attack_method}_{epsilon}_5single_vector_norm{t}{isperb_image}{data_size}.npy'
							# path = '/mnt/cephfs/home/zhangshuhai/DiffPure/score_diffusion_t_live_det/scores_cleansingle_vector_norm50perb_image360.npy'
							# path_adv = '/mnt/cephfs/home/zhangshuhai/DiffPure/score_diffusion_t_spoof_det/scores_cleansingle_vector_norm50perb_image360.npy'
							path = f'/mnt/cephfs/ec/home/zhangshuhai/DiffPure/score_norm_{dataset}_t_ablation/scores_cleansingle_vector_norm{tn/10}{isperb_image}{data_size}_01.npy'
							path_adv = f'/mnt/cephfs/ec/home/zhangshuhai/DiffPure/score_norm_{dataset}_t_ablation/scores_adv_{attack_method}_{epsilon}_5single_vector_norm{tn/10}{isperb_image}{data_size}_01.npy'
							# path = f'score_diffusion_t_imagenet101/scores_cleansingle_vector_norm{tn}{isperb_image}{data_size}.npy'
							# path_adv = f'score_diffusion_t_imagenet101/scores_adv_{attack_method}_{epsilon}_5single_vector_norm{tn}{isperb_image}{data_size}.npy'
						elif dataset=='cifar':
							# path = 'score_diffusion_t_cifar/scores_clean{data_size}.npy'
							# path_adv = f'score_diffusion_t_{dataset}/scores_adv_{attack_method}_{epsilon}{data_size}.npy'
							# path = f'/mnt/cephfs/ec/home/zhangshuhai/score_diffusion_t_cifar/scores_clean200{isperb_image}{data_size}.npy'
							# path_adv = f'score_diffusion_t_{dataset}_stand/scores_adv_{attack_method}_{epsilon}_5200{isperb_image}{data_size}.npy'
							# path = f'score_norm_cifar_wrn_70_16/scores_clean200{isperb_image}{data_size}.npy'
							# path_adv = f'score_norm_cifar_wrn_70_16/scores_adv_{attack_method}_{epsilon}_5{t}{isperb_image}{data_size}.npy'
							# path = f'size_ablation_{dataset}_N/scores_clean{t}{isperb_image}{data_size}.npy'
							# path_adv = f'size_ablation_{dataset}_N/scores_adv_{attack_method}_{epsilon}_5{t}{isperb_image}{data_size}.npy'
							# path = f'score_norm_{dataset}_t_ablation_0.1/scores_clean20.0{isperb_image}{data_size}.npy'
							# path_adv = f'score_norm_{dataset}_t_ablation_0.1/scores_adv_{attack_method}_{epsilon}_520.0{isperb_image}{data_size}.npy'
							# path = f'score_norm_{dataset}_t_ablation_0.1/scores_clean20.0{isperb_image}{data_size}.npy'
							# path_adv = f'score_norm_{dataset}_t_ablation_0.1/scores_adv_{attack_method}_{epsilon}_520.0{isperb_image}{data_size}.npy'
							# path = f'score_{dataset}_t_ablation/scores_clean1000{isperb_image}{data_size}.npy'
							# path_adv = f'score_{dataset}_t_ablation/scores_adv_{attack_method}_{epsilon}_51000{isperb_image}{data_size}.npy'
							path = f'/mnt/cephfs/dataset/zhangshuhai/ICML2022_diffusion_detection/ceph/score_norm_{dataset}_t_ablation/scores_clean{tn}.0{isperb_image}{data_size}.npy'
							path_adv = f'/mnt/cephfs/dataset/zhangshuhai/ICML2022_diffusion_detection/ceph/score_norm_{dataset}_t_ablation/scores_adv_{attack_method}_{epsilon}_5{tn}.0{isperb_image}{data_size}.npy'

						log_dir = f'./score_diffusion_t_{dataset}/'
						os.makedirs(log_dir, exist_ok=True)
						x = torch.from_numpy(np.load(path))
						x_adv = torch.from_numpy(np.load(path_adv))
						
						print(x.shape)
						print(x_adv.shape)
						# [200, 500,3,32,32]
						if dataset == 'cifar' or len(x.shape)!=4:
							# t_list = torch.linspace(0,20, steps=201)[1:][t0+9:tn+1:10]
							# x = x[t0+9:tn+1,:,:,:,:].mean(0)#.half()
							# x_adv = x_adv[t0+9:tn+1,:,:,:,:].mean(0)#.half()
							x = x[t0:tn,:,:,:,:].mean(0).cuda()#.half()
							x_adv = x_adv[t0:tn,:,:,:,:].mean(0).cuda()#.half()
						
						if dataset == 'imagenet':
							x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)#.half()
							x_adv = F.interpolate(x_adv, size=(32, 32), mode='bilinear', align_corners=False)#.half()


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