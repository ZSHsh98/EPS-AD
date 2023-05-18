import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
from D_net import Discriminator, Discriminator_cifar
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import argparse
from score_dataloader import DatasetNPY
from torch.utils.data import DataLoader, Subset
import seaborn as sns
from sklearn import metrics
from time import time
from pytorchcv.model_provider import get_model as ptcv_get_model
from cauculate_MMD import L2_distance_get, mmd_guassian_bigtensor
from torch.autograd import Variable
from utils_MMD import MMDu, MMD_batch
#from network.resnet_orig import ResNet34, ResNet18
#import progress_bar
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--model_name', default="resnet18")
parser.add_argument("--id", type=int, default=50, help="number of experiment")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--diffusion_t", type=int, default=20, help="50 for imagenet")
parser.add_argument("--feature_dim", type=int, default=300, help="300 for imagenet")
parser.add_argument("--epsilon", type=int, default=2, help="10 for imagenet")
parser.add_argument("--seed", type=int, default=999)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--dataset', default='imagenet', type=str)
parser.add_argument('--sigma0', default=15, type=float, help="0.5 for imagenet")
parser.add_argument('--sigma', default=100, type=float, help="100 for imagenet")
parser.add_argument('--isfull',  action='store_false',)
parser.add_argument('--test_flag',  type=bool,default=False)
parser.add_argument('--resume', '-r', action='store_true',
					help='resume from checkpoint')
args = parser.parse_args()

# set random seed
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
id = args.id
# Data
dataset = args.dataset # 'cifar',  'imagenet'
img_size = 224 if dataset == 'imagenet' else 32
batch_size =200 if dataset == 'imagenet' else 200
perb_image = True
isperb_image = 'perb_image' if perb_image else ''
stand_flag = True
isstand = '_stand' if stand_flag else ''
data_size = ''
t = args.diffusion_t
print('==> Preparing data..')
if 1:
	
	if dataset == 'imagenet':
		path_cln = 'score_diffusion_t_val_live_det/scores_cleansingle_vector_norm50perb_image.npy'
		path_adv1 = 'score_diffusion_t_val_spoof_det/scores_cleansingle_vector_norm50perb_image.npy'
		# path_adv2 = 'score_diffusion_t_imagenet2cifar/scores_adv_FGSM_L2_0.00784_5single_vector_norm50perb_image.npy'
		ref_data = np.load(path_cln)	
		x_adv_train1 = np.load(path_adv1)
		# x_adv_train2 = np.load(path_adv2)
		n_x_adv = x_adv_train1.shape[0]

	else:
		path_cln = f'/mnt/cephfs/ec/home/zhangshuhai/score_diffusion_t_cifar/scores_clean200{isperb_image}{data_size}.npy'
		path_adv1 = f'score_diffusion_t_{dataset}_stand/scores_adv_FGSM_0.00392_5200{isperb_image}{data_size}.npy'
		path_adv2 = f'score_diffusion_t_{dataset}_stand/scores_adv_FGSM_L2_0.00392_5200{isperb_image}{data_size}.npy'
		ref_data = np.load(path_cln)[:,100:,:,:,:]	
		x_adv_train1 = np.load(path_adv1)[:,100:,:,:,:]
		x_adv_train2 = np.load(path_adv2)[:,100:,:,:,:]
		n_x_adv = x_adv_train1.shape[1]


# Model
feature_dim = args.feature_dim #if dataset == 'imagenet' else 50
net = Discriminator(img_size=img_size, feature_dim=feature_dim) if dataset == 'imagenet' else Discriminator_cifar(img_size=img_size, feature_dim=feature_dim)
net = net.cuda()

# Initialize parameters
# epsilonOPT = torch.log(torch.from_numpy(np.random.rand(1) * 10 ** (-10)).to(device, torch.float))
epsilonOPT = torch.log(torch.from_numpy(np.random.rand(1) * 10 ** (-args.epsilon)).to(device, torch.float))
epsilonOPT.requires_grad = True
sigmaOPT = torch.from_numpy(np.ones(1) * np.sqrt(2 * img_size * img_size*args.sigma)).to(device, torch.float)
sigmaOPT.requires_grad = True
sigma0OPT = torch.from_numpy(np.ones(1) * np.sqrt(args.sigma0)).to(device, torch.float)
sigma0OPT.requires_grad = True

sigma, sigma0_u, ep = None, None, None
# if args.resume:
# 	# Load checkpoint.
# 	print('==> Resuming from checkpoint..')
# 	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
# 	checkpoint = torch.load('./checkpoint/ckpt.pth')
# 	net.load_state_dict(checkpoint['net'])
# 	best_acc = checkpoint['acc']
# 	start_epoch = checkpoint['epoch']


# if device == 'cuda':
# 	# net = torch.nn.DataParallel(net).cuda()
# 	cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(net.parameters())+ [epsilonOPT] + [sigmaOPT] + [sigma0OPT], lr=args.lr)
epochs = args.epochs
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

	return "auroc: {:.4f}; ".format(ap) + "; ".join(["TPR: {:.4f} @ FPR={:.4f}".format(v, k) for k, v in accs.items()]) + "  {}-{}".format(len(mi_nat), len(mi_svhn))


def train(epoch,ref_data, x_adv_train1):
	print('\nEpoch: %d' % epoch)
	
	net.train()

	# train_loss = 0
	# correct = 0
	# total = 0

	# for batch_idx, (inputs,x_adv1, x_adv2) in enumerate(zip(ref_loader, x_adv_train1, x_adv_train2)):
	# 	# if batch_idx>4:
	# 	# 	break
	# 	if  inputs.shape[0]!=x_adv1.shape[0] or x_adv1.shape[0] != x_adv2.shape[0]:
	# 		break
	# 	inputs = inputs.cuda(non_blocking=True)
	# 	x_adv1= x_adv1[:inputs.shape[0]//2,:,:,:].cuda(non_blocking=True)
	# 	x_adv2= x_adv2[:inputs.shape[0]//2,:,:,:].cuda(non_blocking=True)
	# 	x_adv = torch.cat([x_adv1,x_adv2],dim=0)
	# 	assert inputs.shape[0]==x_adv.shape[0]

	# for batch_idx, (inputs,x_adv) in enumerate(zip(ref_loader, adv_data_loader1)):
	# 	assert inputs.shape[0]==x_adv.shape[0]
	# 	inputs = inputs.cuda(non_blocking=True)
	# 	x_adv = x_adv.cuda(non_blocking=True)
	
	if 1:

		
		ind1 = np.random.choice(n_x_adv, n_x_adv-50, replace=False)
		# ind2 = np.random.choice(n_x_adv, (n_x_adv)//2, replace=False)
		

		if dataset == 'cifar':
			ref_data = ref_data[:t,:,:,:,:].mean(0)
			x_adv_train1 = x_adv_train1[:t,:,:,:,:].mean(0)
			# x_adv_train2 = x_adv_train2[:t,:,:,:,:].mean(0)

		# inputs = Variable(inputs.type(Tensor))
		# x_adv = Variable(x_adv.type(Tensor))
		# inputs = torch.from_numpy(ref_data[ind0]).cuda()
		inputs = torch.from_numpy(ref_data[ind1]).cuda()
		x_adv = torch.from_numpy(x_adv_train1[ind1]).cuda()
		# x_adv2 = torch.from_numpy(x_adv_train2[ind2]).cuda()
		# x_adv = torch.cat([x_adv1,x_adv2],dim=0)


		# print("inputs.shape,x_adv.shape", inputs.shape,x_adv.shape)

		X = torch.cat([inputs, x_adv],dim=0)
		# Y = torch.cat([torch.ones(inputs.shape[0]),torch.zeros(x_adv.shape[0])],dim=0).cuda()

		# valid = Variable(Tensor(inputs.shape[0], 1).fill_(1.0), requires_grad=False)
		# fake = Variable(Tensor(x_adv.shape[0], 1).fill_(0.0), requires_grad=False)
		# Y = torch.cat([valid, fake], 0).squeeze().long()

		optimizer.zero_grad()
		_, outputs = net(X,out_feature=True)
		

		ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
		sigma = sigmaOPT ** 2
		sigma0_u = sigma0OPT ** 2
		# Compute Compute J (STAT_u)
		TEMP = MMDu(outputs, inputs.shape[0], X.view(X.shape[0],-1), sigma, sigma0_u, ep   )
		mmd_value_temp = -1 * (TEMP[0])
		mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
		STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
		# print("STAT_u: ", STAT_u)
		
		# Compute gradient
		STAT_u.backward()

		# Update weights using gradient descent
		optimizer.step()
	
		# print("net.parameters())[-1]", list(net.parameters())[-1])
		# print("net.parameters())[-1]", net.feature.weight.data[:10, 0])
		# print("net.parameters())[-1].grad", net.feature.weight.grad[:10, 0])
	# 	train_loss += mmd_value_temp.item()
	# 	_, predicted = outputs.max(1)
	# 	total += outputs.size(0)
	# 	correct += predicted.eq(Y).sum().item()

	# print(epoch, 'Loss: %.6f | Acc: %.6f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
	print(f"epoch:{epoch}, mmd_value_temp:{mmd_value_temp.item()}, STAT_u:{STAT_u.item()}, mmd_std:{mmd_std_temp}")
	return sigma, sigma0_u, ep

def test(epoch, diffusion_t, dataset):
	global best_acc
	net.eval()
	tt = diffusion_t
	# test_loss = 0
	# correct = 0
	# total = 0
	# value_factor = 15
	# kernel_num = 30
	# fix_sigma = None
	#   'FGSM', 'FGSM_L2','BIM','TIM','BIM_L2' ] #
	attack_methods=['FGSM']#'PGD','FGSM', 'BIM', 'MIM', 'TIM', 'DI_MIM','CW', 'PGD_L2', 'FGSM_L2', 'BIM_L2', 'MM_Attack', 'AA_Attack']
	# attack_method = 'PGD'
	dataset = dataset # 'cifar',  'imagenet', 'imagenet101'
	perb_image = True
	isperb_image = 'perb_image' if perb_image else ''
	stand_flag = True
	isstand = '_stand' if stand_flag else ''
	# t = 50
	for num_sub in [500]:
		data_size = '' if num_sub==500 else str(num_sub)
		for epsilon in [0.00784]: #0.00392, 0.00784, 0.01176, 0.01569, 0.01961, 0.02353, 0.02745, 0.03137]:
			print('dataset:',dataset, 'epsilon:', epsilon)
			for attack_method in attack_methods:
				print(f"======attack_method: {attack_method}")
				for t in [tt]: #20, 50, 100, 150, 200]:
					tile_name = f'scores_face_detect_clean_adv_{attack_method}_{epsilon}_5_{t}{isperb_image}'
					if dataset == 'imagenet':
						# seen attack
						# path_cln = f'score_diffusion_t_{dataset}{isstand}/scores_cleansingle_vector_norm{t}{isperb_image}{data_size}.npy'
						# path_adv = f'score_diffusion_t_{dataset}{isstand}/scores_adv_{attack_method}_{epsilon}_5single_vector_norm{t}{isperb_image}{data_size}.npy'
						# deit-s
						# path_cln = f'score_diffusion_t_deit-s/scores_cleansingle_vector_norm{t}{isperb_image}{data_size}.npy'
						# path_adv = f'score_diffusion_t_deit-s/scores_adv_{attack_method}_{epsilon}_5single_vector_norm{t}{isperb_image}{data_size}.npy'
						# transferable
						# path_cln = f'score_diffusion_t_imagenet101/scores_cleansingle_vector_norm{t}{isperb_image}{data_size}.npy'
						# path_adv = f'score_diffusion_t_imagenet101/scores_adv_{attack_method}_{epsilon}_5single_vector_norm{t}{isperb_image}{data_size}.npy'
						# t _ablation
						# path_cln = f'score_norm_{dataset}_t_ablation/scores_cleansingle_vector_norm{t}{isperb_image}{data_size}.npy'
						# path_adv = f'score_norm_{dataset}_t_ablation/scores_adv_{attack_method}_{epsilon}_5single_vector_norm{t}{isperb_image}{data_size}.npy'
						# size_ablation
						# path_cln = f'size_ablation_resnet50_N/scores_cleansingle_vector_norm50{isperb_image}{data_size}.npy'
						# path_adv = f'size_ablation_resnet50_N/scores_adv_{attack_method}_{epsilon}_5single_vector_norm50{isperb_image}{data_size}.npy'
						# face
						path_cln = f'score_diffusion_t_test_live_det/scores_cleansingle_vector_norm50perb_image480.npy'
						path_adv = f'score_diffusion_t_test_spoof_det/scores_cleansingle_vector_norm50perb_image1920.npy'
						
					else:
						# seen attack
						path_cln = f'/mnt/cephfs/ec/home/zhangshuhai/score_diffusion_t_cifar/scores_clean200{isperb_image}{data_size}.npy'
						path_adv = f'score_diffusion_t_{dataset}_stand/scores_adv_{attack_method}_{epsilon}_5200{isperb_image}{data_size}.npy'
						# wrn-70-16
						# path_cln = f'score_norm_{dataset}_wrn_70_16/scores_clean200{isperb_image}{data_size}.npy'
						# path_adv = f'score_norm_{dataset}_wrn_70_16/scores_adv_{attack_method}_{epsilon}_520{isperb_image}{data_size}.npy'
						# t_ablation ?
						# path_cln = f'score_norm_cifar_t_ablation/scores_clean_norm{t}.0{isperb_image}{data_size}.npy'
						# path_adv = f'score_norm_cifar_t_ablation/scores_adv_{attack_method}_{epsilon}_5_norm{t}.0{isperb_image}{data_size}.npy'
						# size_ablation
						# path_cln = f'size_ablation_{dataset}_ours/scores_clean20{isperb_image}{data_size}.npy'
						# path_adv = f'size_ablation_{dataset}_ours/scores_adv_{attack_method}_{epsilon}_520{isperb_image}{data_size}.npy'
					log_dir = f'score_diffusion_t_face_detect_{dataset}{isstand}/test/'
					os.makedirs(log_dir, exist_ok=True)
					with torch.no_grad():
						# ref_list = []
						# for batch_idx, (inputs) in enumerate(ref_loader):
						# 	if batch_idx>4:
						# 		break
						# 	ref_list.append(inputs)
						
						# ref_data = torch.cat(ref_list,dim=0).cuda()
						# ref_data = torch.from_numpy(np.load(path_cln)[100:]).cuda()
						ref_data = torch.from_numpy(x_adv_train1).cuda()
						x_cln = torch.from_numpy(np.load(path_cln)[:100]).cuda()
						x_adv = torch.from_numpy(np.load(path_adv)[:100]).cuda()
						# print(x_cln.shape)
						# print(x_adv.shape)
						
						if dataset == 'cifar':
							ref_data = ref_data[:t,:,:,:,:].mean(0)#.half() 
							x_cln = x_cln[:t,:,:,:,:].mean(0)#.half() 
							x_adv = x_adv[:t,:,:,:,:].mean(0)#.half()

						_,feature_ref = net(ref_data,out_feature=True)
						_,feature_cln = net(x_cln,out_feature=True)
						_,feature_adv = net(x_adv,out_feature=True)

						dt_clean = MMD_batch(torch.cat([feature_ref,feature_cln],dim=0), feature_ref.shape[0], torch.cat([ref_data,x_cln],dim=0).view(ref_data.shape[0]+x_cln.shape[0],-1), sigma, sigma0_u, ep).cpu()
						dt_adv = MMD_batch(torch.cat([feature_ref,feature_adv],dim=0), feature_ref.shape[0], torch.cat([ref_data,x_adv],dim=0).view(ref_data.shape[0]+x_adv.shape[0],-1), sigma, sigma0_u, ep).cpu()
						
						# L2_distance_xx = L2_distance_get(feature_ref,feature_ref, value_factor)
						# L2_distance_yx_cln = L2_distance_get(feature_cln, feature_ref, value_factor)
						# L2_distance_yx_adv = L2_distance_get(feature_adv, feature_ref, value_factor)

						# dt_adv = mmd_guassian_bigtensor_batch(L2_distance_xx, L2_distance_yx_adv,value_factor,kernel_num,fix_sigma=fix_sigma).cpu()
						# dt_clean = mmd_guassian_bigtensor_batch(L2_distance_xx, L2_distance_yx_cln,value_factor,kernel_num,fix_sigma=fix_sigma).cpu()
						
						print('t: '+ str(t))
						print(plot_mi( dt_clean, dt_adv,log_dir, tile_name))

	if not args.test_flag:
		model_path = f'./net_D/face/{id}' 
		state = {
				'net': net.state_dict(),
				'epsilonOPT': epsilonOPT,
				'sigmaOPT': sigmaOPT,
				'sigma0OPT': sigma0OPT,
				'sigma': sigma,
				'sigma0_u':sigma0_u,
				'ep': ep
			}
		if not os.path.isdir(model_path):
			os.makedirs(model_path, exist_ok=True)
			# os.mkdir(model_path)
		if (epoch+1)%100==0:
			torch.save(state, model_path + '/'+ str(epoch) +'_ckpt.pth')
		torch.save(state, model_path + '/'+ 'last_ckpt.pth')

if not args.test_flag:
	for epoch in range(start_epoch, start_epoch+epochs):
		time0 = time()
		sigma, sigma0_u, ep =train(epoch,ref_data, x_adv_train1)
		print("time:",time()-time0,"epoch",epoch, "sigma, sigma0_u, ep", sigma, sigma0_u, ep)
		if (epoch+1)%20==0:
			test(epoch, t, dataset)
else:
	epoch = 99
	print('==> testing from checkpoint..')
	model_path = f'./net_D/face/{id}' 
	assert os.path.isdir(model_path), 'Error: no checkpoint directory found!'
	# checkpoint = torch.load(model_path + '/'+ str(epoch) +'last_ckpt.pth')
	checkpoint = torch.load(model_path + '/'+ 'last_ckpt.pth')
	net.load_state_dict(checkpoint['net'])
	sigma, sigma0_u, ep  = checkpoint['sigma'], checkpoint['sigma0_u'], checkpoint['ep']
	test(epoch, t, dataset)


# optimal paramater setup
# imagenet
# CUDA_VISIBLE_DEVICES=4 python train_D_extractor_2_face.py --epochs 200 --lr 0.002 --id 50 --sigma0 0.5 --sigma 100  --epsilon 10 --feature_dim 300 --dataset imagenet --diffusion_t 50