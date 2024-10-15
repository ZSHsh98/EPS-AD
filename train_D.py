from sqlalchemy import false
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
from D_net import Discriminator, Discriminator_cifar, Discriminator_cifar2
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '7'
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
parser.add_argument("--id", type=int, default=999, help="number of experiment")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--feature_dim", type=int, default=300, help="300 for imagenet")
parser.add_argument("--epsilon", type=int, default=10, help="10 for imagenet")
parser.add_argument("--seed", type=int, default=999)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--dataset', default='imagenet', type=str)
parser.add_argument('--sigma0', default=0.5, type=float, help="0.5 for imagenet")
parser.add_argument('--sigma', default=100, type=float, help="100 for imagenet")
parser.add_argument('--isfull',  action='store_false',)
parser.add_argument('--test_flag',  type=bool,default=False)
parser.add_argument('--detection_datapath', type=str, default='./')
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
batch_size =200 if dataset == 'imagenet' else 500
SIZE = 500
perb_image = True
isperb_image = 'perb_image' if perb_image else ''
stand_flag = True
isstand = '_stand' if stand_flag else ''
data_size = ''
t = 50 if dataset == 'imagenet' else 20
datapath = f'{args.detection_datapath}/score_diffusion_t_{dataset}_1w'

print('==> Preparing data..')

path = f'{datapath}/scores_cleansingle_vector_norm{t}perb_image10000/'
ref_data = DatasetNPY(path)
ref_loader = DataLoader(ref_data, batch_size=batch_size, shuffle=True, num_workers=8)

if not args.test_flag:
	path_adv = f'{datapath}/scores_adv_FGSM_L2_0.00392_5single_vector_norm{t}perb_image10000/'
	adv_data1 = DatasetNPY(path_adv)
	adv_data_loader1 = DataLoader(adv_data1, batch_size=batch_size, shuffle=True, num_workers=8)

	path_adv2 = f'{datapath}/scores_adv_FGSM_0.00392_5single_vector_norm{t}perb_image10000/'
	adv_data2 = DatasetNPY(path_adv2)
	adv_data_loader2 = DataLoader(adv_data2, batch_size=batch_size, shuffle=True, num_workers=8)

if '128' in path:
	img_size = 128

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


def train(epoch):
	print('\nEpoch: %d' % epoch)
	
	net.train()

	# train_loss = 0
	# correct = 0
	# total = 0

	for batch_idx, (inputs,x_adv1, x_adv2) in enumerate(zip(ref_loader, adv_data_loader1, adv_data_loader2)):
		if batch_idx>30:
			break
		if inputs.shape[0]!=x_adv1.shape[0] or x_adv1.shape[0] != x_adv2.shape[0]:
			break
		# assert inputs.shape[0]==x_adv1.shape[0] and x_adv1.shape[0] == x_adv2.shape[0]
		inputs = inputs.cuda(non_blocking=True)
		x_adv1= x_adv1[:inputs.shape[0]//2,:,:,:].cuda(non_blocking=True)
		x_adv2= x_adv2[:inputs.shape[0]//2,:,:,:].cuda(non_blocking=True)
		x_adv = torch.cat([x_adv1,x_adv2],dim=0)
		assert inputs.shape[0]==x_adv.shape[0]


		X = torch.cat([inputs, x_adv],dim=0)


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

	print(f"epoch:{epoch}, mmd_value_temp:{mmd_value_temp.item()}, STAT_u:{STAT_u.item()}, mmd_std:{mmd_std_temp}")
	return sigma, sigma0_u, ep

def test(epoch, diffusion_t, dataset):
	global best_acc
	net.eval()
	tt = diffusion_t

	# attack_methods=['PGD','FGSM', 'BIM', 'MIM', 'TIM', 'DI_MIM','CW', 'PGD_L2', 'FGSM_L2', 'BIM_L2', 'MM_Attack', 'AA_Attack', 'VMI_FGSM']
	attack_methods=['FGSM_L2']#,'MIM', 'TIM', 'DI_MIM','BIM_L2', 'MM_Attack']
	# attack_method = 'PGD'
	dataset = dataset # 'cifar',  'imagenet', 'imagenet101'
	perb_image = True
	isperb_image = 'perb_image' if perb_image else ''
	stand_flag = True
	isstand = '_stand' if stand_flag else ''
	for num_sub in [500]:
		data_size = '' if num_sub==500 else str(num_sub)
		for epsilon in [0.01569]: #0.00392, 0.00784, 0.01176, 0.01569, 0.01961, 0.02353, 0.02745, 0.03137]:
			print('dataset:',dataset, 'epsilon:', epsilon)
			for attack_method in attack_methods:
				print(f"======attack_method: {attack_method}")
				# for t in [50]:
				if 1:
					tile_name = f'scores_face_detect_clean_adv_{attack_method}_{epsilon}_5_{t}{isperb_image}'

					path_cln = f'{args.detection_datapath}/score_diffusion_t_{dataset}_stand/scores_cleansingle_vector_norm{t}{isperb_image}{data_size}.npy'
					path_adv = f'{args.detection_datapath}/score_diffusion_t_{dataset}_stand/scores_adv_{attack_method}_{epsilon}_5single_vector_norm{t}{isperb_image}{data_size}.npy'
					
					log_dir = f'{args.detection_datapath}/score_diffusion_detect_{dataset}{isstand}/test/'
					os.makedirs(log_dir, exist_ok=True)
					with torch.no_grad():
						ref_list = []
						for batch_idx, (inputs) in enumerate(ref_loader):
							if batch_idx>3:
								break
							ref_list.append(inputs)
						
						ref_data = torch.cat(ref_list,dim=0).cuda()[:SIZE]
						x_cln = torch.from_numpy(np.load(path_cln)).cuda()
						x_adv = torch.from_numpy(np.load(path_adv)).cuda()
						
						
						time0 = time()
						_,feature_ref = net(ref_data,out_feature=True)
						_,feature_cln = net(x_cln,out_feature=True)
						_,feature_adv = net(x_adv,out_feature=True)


						dt_clean = MMD_batch(torch.cat([feature_ref,feature_cln],dim=0), feature_ref.shape[0], torch.cat([ref_data,x_cln],dim=0).view(ref_data.shape[0]+x_cln.shape[0],-1), sigma, sigma0_u, ep).cpu()
						dt_adv = MMD_batch(torch.cat([feature_ref,feature_adv],dim=0), feature_ref.shape[0], torch.cat([ref_data,x_adv],dim=0).view(ref_data.shape[0]+x_adv.shape[0],-1), sigma, sigma0_u, ep).cpu()
						
						print(plot_mi( dt_clean, dt_adv,log_dir, tile_name))

	if not args.test_flag:
		model_path = f'./net_D/{args.dataset}/{id}' 
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
		sigma, sigma0_u, ep =train(epoch)
		print("time:",time()-time0,"epoch",epoch, "sigma, sigma0_u, ep", sigma, sigma0_u, ep)
		if (epoch+1)%20==0:
			test(epoch, t, dataset)
else:
	epoch = 99
	print('==> testing from checkpoint..')
	model_path = f'./net_D/{args.dataset}/{id}'
	assert os.path.isdir(model_path), 'Error: no checkpoint directory found!'
	# checkpoint = torch.load(model_path + '/'+ str(epoch) +'last_ckpt.pth')
	checkpoint = torch.load(model_path + '/'+ 'last_ckpt.pth')
	net.load_state_dict(checkpoint['net'])
	sigma, sigma0_u, ep  = checkpoint['sigma'], checkpoint['sigma0_u'], checkpoint['ep']
	test(epoch, t, dataset)


# optimal paramater setup
# cifar
# CUDA_VISIBLE_DEVICES=6 python train_D_extractor_2.py --epochs 200 --lr 0.00002 --id 8 --sigma0 15 --sigma 100  --epsilon 2 --feature_dim 300 --dataset cifar
# imagenet
# CUDA_VISIBLE_DEVICES=7 python train_D_extractor_2.py --epochs 200 --lr 0.002 --id 6 --sigma0 0.5 --sigma 100  --epsilon 10 --feature_dim 300 --dataset imagenet