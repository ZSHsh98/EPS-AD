# from eagerpy import tile
# from bleach import clean
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import os 
import torch
import seaborn as sns
from sklearn import metrics
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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

attack_method = 'PGD'
t = 100
# scores_f_fg [1000,500,3,32,32]
path = 'score_diffusion_t_cifar/scores_clean.npy'
path_adv = f'score_diffusion_t_cifar/scores_adv_{attack_method}_0.01568_5.npy'
tile_name = f'scores_mmd_ref_mean_clean_adv_{attack_method}_0.01568_5_{t}'
log_dir = './score_diffusion_t_cifar/'
os.makedirs(log_dir, exist_ok=True)
x = torch.from_numpy(np.load(path))
x_adv = torch.from_numpy(np.load(path_adv))

'''##### calculate the mean by the dimention of samples 
# 1) calculate bar_xt [1000, 1, 3, 32, 32]
bar_xt = x.mean(1, keepdims=True)
# 2) calculate distance dt [1000, 500]
dt_clean = (x - bar_xt).view(x.shape[0]*x.shape[1],-1).norm(dim=-1).view(x.shape[:2])
dt_adv = (x_adv - bar_xt).view(x.shape[0]*x.shape[1],-1).norm(dim=-1).view(x.shape[:2])

# a = 0
# b = 20
# num_samples = dt_clean.shape[1]
# for i in range(num_samples):
#     clean = dt_clean[a:b,i] 
#     adv = dt_adv[a:b,i]

#     plt.plot(clean, color='red')
#     plt.plot(adv, color='blue')
#     plt.title(f'scores_t_{a}-{b}')
#     plt.savefig(f'{log_dir}/scores_var_vector_clean_adv_t_{a}-{b}.jpg')
# plt.close()

sum_dt_clean = dt_clean[0:20,:].sum(0)
sum_dt_adv = dt_adv[0:20,:].sum(0)
print(plot_mi(sum_dt_clean, sum_dt_adv, log_dir))
num_bins = 50
plt.hist(sum_dt_clean, num_bins, facecolor='red', density=True, alpha=0.9)
plt.hist(sum_dt_adv, num_bins,  facecolor='blue', density=True,alpha=0.9)
plt.title(f'scores_var_vector_clean_adv_sum_20')
plt.savefig(f'{log_dir}/scores_var_vector_clean_adv_sum_20.jpg')
plt.close()'''

'''############ calculate the mean by the dimention of time 
# 1) calculate bar_x_sigle_sample [1, 500, 3, 32, 32]
bar_x_sigle_sample = x[:100,:,:,:,:].mean(0, keepdims=True)
bar_x_adv_sigle_sample = x_adv[:100,:,:,:,:].mean(0, keepdims=True)
# 2) calculate distance dt [500]
dt_clean = bar_x_sigle_sample.view(x.shape[1],-1).norm(dim=-1).view(-1)
dt_adv = bar_x_adv_sigle_sample.view(x_adv.shape[1],-1).norm(dim=-1).view(-1)

tile_name = 'scores_var_vector_clean_adv_eps0.0313_iter5_sum_100'
print(plot_mi(dt_clean, dt_adv, log_dir, tile_name))
num_bins = 50
plt.hist(dt_clean, num_bins, facecolor='red', density=True, alpha=0.9)
plt.hist(dt_adv, num_bins,  facecolor='blue', density=True,alpha=0.9)
plt.title(f'{tile_name}')
plt.savefig(f'{log_dir}/{tile_name}.jpg')
plt.close()'''


'''############  centralization
# 1) calculate bar_x_sigle_sample [1, 500, 3, 32, 32]
bar_x_sigle_sample = x.mean(0, keepdims=True)
bar_x_adv_sigle_sample = x_adv.mean(0, keepdims=True)

bar_x_sigle = bar_x_sigle_sample.mean(1,keepdims=True)
# 2) calculate distance dt [500]
dt_clean = (bar_x_sigle_sample - bar_x_sigle).view(x.shape[1],-1).norm(dim=-1).view(-1)
dt_adv = (bar_x_adv_sigle_sample - bar_x_sigle).view(x_adv.shape[1],-1).norm(dim=-1).view(-1)

print(plot_mi(dt_clean, dt_adv, log_dir, 'scores_var_vector'))
num_bins = 50
plt.hist(dt_clean, num_bins, facecolor='red', density=True, alpha=0.9)
plt.hist(dt_adv, num_bins,  facecolor='blue', density=True,alpha=0.9)
plt.title(f'scores_var_vector')
plt.savefig(f'{log_dir}/scores_var_vector.jpg')
plt.close()'''

'''############ difference with adjacent time
diff_x_clean = x[1:1000,:,:,:,:] - x[0:999,:,:,:,:]
diff_x_adv = x_adv[1:1000,:,:,:,:] - x_adv[0:999,:,:,:,:]
# sum_diff [500]
sum_diff_x_clean = diff_x_clean.view(*diff_x_clean.shape[:2],-1).norm(dim=-1).sum(0)/diff_x_clean.shape[0] 
sum_diff_x_adv = diff_x_adv.view(*diff_x_adv.shape[:2],-1).norm(dim=-1).sum(0)/diff_x_clean.shape[0]
tile_name = 'scores_t_adjacent_diff'
print(plot_mi(sum_diff_x_clean, sum_diff_x_adv, log_dir, tile_name))
num_bins = 50
plt.hist(sum_diff_x_clean, num_bins, facecolor='red', density=True, alpha=0.9)
plt.hist(sum_diff_x_adv, num_bins,  facecolor='blue', density=True,alpha=0.9)
plt.title(f'{tile_name}')
plt.savefig(f'{log_dir}/{tile_name}.jpg')
plt.close()'''

############ calculate the refernce vetor by the dimention of samples via norm
# 1) calculate bar_x_norm [1000,1]
# bar_x_norm = x.view(x.shape[0], x.shape[1], -1).norm(dim=-1).mean(1, keepdims=True)
bar_x_norm = x.view(x.shape[0], x.shape[1], -1).norm(dim=-1) #[1000, 500]
# 2) calculate mmd distance dt [500]
x_norm = x.view(x.shape[0], x.shape[1], -1).norm(dim=-1) #[1000, 500]
x_adv_norm = x_adv.view(x_adv.shape[0], x_adv.shape[1], -1).norm(dim=-1) #[1000, 500]
from cauculate_MMD import mmd
def dis_ref_clean2data(ref_clean, data):
	print(f"The dimension of feature is {ref_clean.shape[0]}")
	assert len(ref_clean.shape)==2 and len(data.shape)==2
	dis_vector = torch.zeros_like(data[0],device=data.device)
	for i in range(data.shape[1]):
		dis_vector[i] = mmd(ref_clean.transpose(0,1),data[:,i].unsqueeze(0))
	return dis_vector

dt_clean = dis_ref_clean2data(bar_x_norm[:t,:], x_norm[:t,:])
dt_adv = dis_ref_clean2data(bar_x_norm[:t,:], x_adv_norm[:t,:])

print(plot_mi(dt_clean, dt_adv, log_dir, tile_name))
num_bins = 50
plt.hist(dt_clean, num_bins, facecolor='red', density=True, alpha=0.9)
plt.hist(dt_adv, num_bins,  facecolor='blue', density=True,alpha=0.9)
plt.title(f'{tile_name}')
plt.savefig(f'{log_dir}/{tile_name}.jpg')
plt.close()