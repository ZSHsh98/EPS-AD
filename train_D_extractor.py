import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
from D_net import Discriminator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import argparse
from score_dataloader import DatasetNPY
from torch.utils.data import DataLoader
import seaborn as sns
from sklearn import metrics
from time import time
from pytorchcv.model_provider import get_model as ptcv_get_model
from cauculate_MMD import L2_distance_get, mmd_guassian_bigtensor
from torch.autograd import Variable
#from network.resnet_orig import ResNet34, ResNet18
#import progress_bar
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--model_name', default="resnet18")
parser.add_argument("--id", type=int, default=1000, help="number of experiment")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--seed", type=int, default=999)
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
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
mean_ref, var_ref = 0., 1.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
id = args.id
# Data
dataset = 'imagenet' # 'cifar',  'imagenet'
batch_size =200
perb_image = True
isperb_image = 'perb_image' if perb_image else ''
stand_flag = True
isstand = '_stand' if stand_flag else ''
data_size = ''
t = 50
print('==> Preparing data..')
if dataset=='imagenet':
    path = f'/mnt/cephfs/ec/home/zhangshuhai/score_diffusion_t_{dataset}/scores_cleansingle_vector_norm{t}{isperb_image}10000/'
ref_data = DatasetNPY(path)
ref_loader = DataLoader(ref_data, batch_size=batch_size, shuffle=True, num_workers=8)

path_adv = f'/mnt/cephfs/ec/home/zhangshuhai/score_diffusion_t_coco//scores_cleansingle_vector_norm50{isperb_image}{data_size}.npy'
# path_adv = f'score_diffusion_t_imagenet_stand/scores_adv_BIM_L2_0.01569_5single_vector_norm50perb_image.npy'
x_adv_train = np.load(path_adv)
n_x_adv = x_adv_train.shape[0]


# Model
net = Discriminator()
net = net.cuda()


# if args.test_flag:
	# Load checkpoint.

# 	best_acc = checkpoint['acc']
# 	start_epoch = checkpoint['epoch']


# if device == 'cuda':
# 	# net = torch.nn.DataParallel(net).cuda()
# 	cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
epochs = args.epochs
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

def adjust_learning_rate(optimizer, epoch):
	if epoch < 200:#80:
		lr = 0.1
	elif epoch < 300:#120:
		lr = 0.01
	else:
		lr = 0.001
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def mmd_guassian_bigtensor_batch(L2_distance_xx, L2_distance_yx_cln,value_factor,kernel_num, fix_sigma=None):
	dis_vector = torch.zeros_like(L2_distance_yx_cln[:,0],device=L2_distance_yx_cln.device)
	for i in range(L2_distance_yx_cln.shape[0]):
		dis_vector[i] = mmd_guassian_bigtensor(L2_distance_xx, L2_distance_yx_cln[i].unsqueeze(0), value_factor=value_factor, kernel_num=kernel_num, fix_sigma=fix_sigma)
	return dis_vector

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

# Training
def train(epoch):
	print('\nEpoch: %d' % epoch)
	
	net.train()

	train_loss = 0
	correct = 0
	total = 0
	#adjust_learning_rate(optimizer, epoch)
	for batch_idx, (inputs) in enumerate(ref_loader):
		#inputs, targets = inputs.to(device), targets.to(device)
		inputs = inputs.cuda(non_blocking=True)
		ind = np.random.choice(n_x_adv, inputs.shape[0], replace=False)
		x_adv = torch.from_numpy(x_adv_train[ind]).cuda()

		X = torch.cat([inputs, x_adv],dim=0)
		# Y = torch.cat([torch.ones(inputs.shape[0]),torch.zeros(x_adv.shape[0])],dim=0).cuda()

		valid = Variable(Tensor(inputs.shape[0], 1).fill_(1.0), requires_grad=False)
		fake = Variable(Tensor(x_adv.shape[0], 1).fill_(0.0), requires_grad=False)
		Y = torch.cat([valid, fake], 0).squeeze().long()

		optimizer.zero_grad()
		outputs = net(X)
		loss = criterion(outputs, Y)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += outputs.size(0)
		correct += predicted.eq(Y).sum().item()

	print(epoch, 'Loss: %.6f | Acc: %.6f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, save_last_flag=True):
	global best_acc
	net.eval()
	# test_loss = 0
	# correct = 0
	# total = 0
	value_factor = 15
	kernel_num = 30
	fix_sigma = None
	attack_methods=[ 'BIM','TIM', 'FGSM_L2', 'BIM_L2', 'AA_Attack'] #'PGD','FGSM', 'BIM', 'MIM', 'TIM', 'DI_MIM','CW', 'PGD_L2', 'FGSM_L2', 'BIM_L2', 'MM_Attack', 'AA_Attack']
	# attack_method = 'PGD'
	dataset = 'imagenet' # 'cifar',  'imagenet'
	perb_image = True
	isperb_image = 'perb_image' if perb_image else ''
	stand_flag = True
	isstand = '_stand' if stand_flag else ''
	t = 50
	# mean_ref, var_ref = mean_var_for_dataset(ref_loader)
	for num_sub in [500]:
		data_size = '' if num_sub==500 else str(num_sub)
		for epsilon in [0.01569]: #0.00392, 0.00784, 0.01176, 0.01569, 0.01961, 0.02353, 0.02745, 0.03137]:
			print('dataset:',dataset, 'epsilon:', epsilon)
			for attack_method in attack_methods:
				tile_name = f'scores_face_detect_clean_adv_{attack_method}_{epsilon}_5_{t}{isperb_image}'
				print(f"======attack_method: {attack_method}")
				for t in [50]: #20, 50, 100, 150, 200]:
					path_cln = f'score_diffusion_t_{dataset}{isstand}/scores_cleansingle_vector_norm{t}{isperb_image}{data_size}.npy'
					path_adv = f'score_diffusion_t_{dataset}{isstand}/scores_adv_{attack_method}_{epsilon}_5single_vector_norm{t}{isperb_image}{data_size}.npy'
					log_dir = f'score_diffusion_t_face_detect_{dataset}{isstand}/test/'
					os.makedirs(log_dir, exist_ok=True)
					with torch.no_grad():
						ref_list = []
						for batch_idx, (inputs) in enumerate(ref_loader):
							if batch_idx>4:
								break
							ref_list.append(inputs)
						
						ref_data = torch.cat(ref_list,dim=0).cuda()
						x_cln = torch.from_numpy(np.load(path_cln)).cuda()
						x_adv = torch.from_numpy(np.load(path_adv)).cuda()
						
						_,feature_ref = net(ref_data,out_feature=True)
						_,feature_cln = net(x_cln,out_feature=True)
						_,feature_adv = net(x_adv,out_feature=True)
						
						# feature_ref = (feature_ref - mean_ref)/var_ref
						# feature_cln = (feature_cln - mean_ref)/var_ref
						# feature_adv = (feature_adv - mean_ref)/var_ref


						L2_distance_xx = L2_distance_get(feature_ref,feature_ref, value_factor)
						L2_distance_yx_cln = L2_distance_get(feature_cln, feature_ref, value_factor)
						L2_distance_yx_adv = L2_distance_get(feature_adv, feature_ref, value_factor)

						# dt_clean = L2_distance_yx_cln.view(L2_distance_yx_cln.shape[0],-1).sum(dim=-1).cpu()
						# dt_adv = L2_distance_yx_adv.view(L2_distance_yx_adv.shape[0],-1).sum(dim=-1).cpu()

						# print(plot_mi(dt_clean, dt_adv, log_dir, tile_name))
						dt_adv = mmd_guassian_bigtensor_batch(L2_distance_xx, L2_distance_yx_adv,value_factor,kernel_num,fix_sigma=fix_sigma).cpu()
						dt_clean = mmd_guassian_bigtensor_batch(L2_distance_xx, L2_distance_yx_cln,value_factor,kernel_num,fix_sigma=fix_sigma).cpu()
						
						print(plot_mi( dt_clean, dt_adv,log_dir, tile_name))

	model_path = f'./net_D/{id}' 
	if not os.path.isdir(model_path):
		os.mkdir(model_path)
	if save_last_flag and (epoch+1)%100==0:
		torch.save(net.state_dict(), model_path + '/'+ str(epoch) +'_ckpt.pth')
	if save_last_flag:
		torch.save(net.state_dict(), model_path + '/'+ 'last_ckpt.pth')

def mean_var_for_dataset(dataset):
	feature_ref_list = []
	total_sum = 0
	net.eval()
	with torch.no_grad():
		for batch_idx, (inputs) in enumerate(dataset):
			if batch_idx>20:
				break
			#inputs, targets = inputs.to(device), targets.to(device)
			inputs = inputs.cuda(non_blocking=True)
			_,feature_ref = net(inputs,out_feature=True)
			feature_ref_list.append(feature_ref)
		feature_ref_tensor = torch.cat(feature_ref_list,dim=0)
		feature_ref_list.clear()
		print("feature_ref_tensor.shape:",feature_ref_tensor.shape)
		mean_ref = feature_ref_tensor.mean()
		var_ref  = feature_ref_tensor.var()
	return mean_ref, var_ref


if not args.test_flag:
	for epoch in range(start_epoch, start_epoch+epochs):
		time0 = time()
		train(epoch)
		test(epoch)
		print("Time:",time()-time0)
		# scheduler.step()
else:
	epoch = 99
	print('==> testing from checkpoint..')
	model_path = f'./net_D/{id}' 
	assert os.path.isdir(model_path), 'Error: no checkpoint directory found!'
	# checkpoint = torch.load(model_path + '/'+ str(epoch) +'last_ckpt.pth')
	checkpoint = torch.load(model_path + '/'+ 'last_ckpt.pth')
	net.load_state_dict(checkpoint)
	mean_ref, var_ref = mean_var_for_dataset(ref_loader)
	test(epoch, save_last_flag=False)
# CUDA_VISIBLE_DEVICES=6 python train_D_extractor.py --epochs 200 --lr 0.0002 --id 3