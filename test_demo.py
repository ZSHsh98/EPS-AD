import torch
import torch.nn as nn
import random
import numpy as np
# import torchvision.transforms as transforms
from D_net import Discriminator, Discriminator_cifar
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import argparse
from score_dataloader import DatasetNPY
from torch.utils.data import DataLoader
from sklearn import metrics
from utils_MMD import MMD_batch2

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
import joblib

from utils import str2bool, get_image_classifier, load_detection_data
import yaml
import utils
from ensattack import ens_attack
#from network.resnet_orig import ResNet34, ResNet18
#import progress_bar
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--model_name', default="resnet18")
parser.add_argument("--id", type=int, default=999, help="number of experiment")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--feature_dim", type=int, default=300, help="300 for imagenet")
parser.add_argument("--epsilon_MMD", type=int, default=10, help="10 for imagenet")
# parser.add_argument("--seed", type=int, default=999)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--dataset', default='imagenet', type=str)
parser.add_argument('--sigma0', default=0.5, type=float, help="0.5 for imagenet")
parser.add_argument('--sigma', default=100, type=float, help="100 for imagenet")
parser.add_argument('--isfull',  action='store_false',)
parser.add_argument('--test_flag',  type=bool,default=False)
# parser.add_argument('--detection_datapath', type=str, default='./score_diffusion_t_cifar_1w')
parser.add_argument('--resume', '-r', action='store_true',
					help='resume from checkpoint')

# diffusion models
parser.add_argument('--config', type=str, default='cifar10.yml', help='Path to the config file')
parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
parser.add_argument('--seed', type=int, default=1235, help='Random seed')
parser.add_argument('--exp', type=str, default='./exp_results', help='Path for saving running related data.')
parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
parser.add_argument('-i', '--image_folder', type=str, default='cifar10', help="The folder name of samples")
parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
parser.add_argument('--t', type=int, default=1000, help='Sampling noise scale')
parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
parser.add_argument('--diffusion_type', type=str, default='sde', help='[ddpm, sde]')
parser.add_argument('--score_type', type=str, default='score_sde', help='[guided_diffusion, score_sde]')
parser.add_argument('--eot_iter', type=int, default=20, help='only for rand version of autoattack')
parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')
parser.add_argument('--datapath', type=str, default='./dataset')

# Detection
parser.add_argument('--clean_score_flag', action='store_true')
parser.add_argument('--detection_datapath', type=str, default='./score_diffusion_t_cifar')#./score_diffusion_t_cifar
# parser.add_argument('--detection_flag', action='store_true')
# parser.add_argument('--detection_ensattack_flag', action='store_true')
parser.add_argument('--detection_ensattack_norm_flag', action='store_true')
parser.add_argument('--generate_1w_flag', action='store_true')
# parser.add_argument('--single_vector_norm_flag', action='store_true')	
parser.add_argument('--t_size', type=int,default=10)
parser.add_argument('--diffuse_t', type=int,default=100)
parser.add_argument('--perb_image', action='store_true')
parser.add_argument('--loader', default=None, help='your preferred data loader')	

# LDSDE
parser.add_argument('--sigma2', type=float, default=1e-3, help='LDSDE sigma2')
parser.add_argument('--lambda_ld', type=float, default=1e-2, help='lambda_ld')
parser.add_argument('--eta', type=float, default=5., help='LDSDE eta')
parser.add_argument('--step_size', type=float, default=1e-2, help='step size for ODE Euler method')

# adv
parser.add_argument('--domain', type=str, default='cifar10', help='which domain: celebahq, cat, car, imagenet')
parser.add_argument('--classifier_name', type=str, default='cifar10-wideresnet-28-10', help='which classifier to use')
parser.add_argument('--partition', type=str, default='val')
parser.add_argument('--adv_batch_size', type=int, default=64)
parser.add_argument('--attack_type', type=str, default='square')
parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'])
parser.add_argument('--attack_version', type=str, default='standard')

# additional attack settings
parser.add_argument('--num-steps', default=5, type=int,help='perturb number of steps')
parser.add_argument('--random', default=True,help='random initialization for PGD')
parser.add_argument('--attack_methods', type=str, nargs='+',default=['FGSM', 'PGD', 'BIM',  'MIM', 'TIM', 'CW', 'DI_MIM','FGSM_L2', 'PGD_L2', 'BIM_L2','MM_Attack', 'AA_Attack'])
parser.add_argument('--mim_momentum', default=1., type=float,help='mim_momentum')
parser.add_argument('--epsilon', default=0.01568, type=float,help='perturbation')#0.01568, type=float,help='perturbation')

parser.add_argument('--num_sub', type=int, default=64, help='imagenet subset')
parser.add_argument('--adv_eps', type=float, default=0.031373, help='0.031373')
parser.add_argument('--gpu_ids', type=str, default='3,4')

args = parser.parse_args()
args.step_size_adv = args.epsilon / args.num_steps

#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
# parse config file
with open(os.path.join('configs', args.config), 'r') as f:
	config = yaml.safe_load(f)
config = utils.dict2namespace(config)

args = parser.parse_args()

# set random seed
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config.device = device
Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor
# best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
id = args.id
# Data
dataset = args.dataset # 'cifar',  'imagenet'
print(f'==> Your refered data is {dataset}..')
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

path = f'{datapath}/scores_cleansingle_vector_norm{t}perb_image1000/'
ref_data = DatasetNPY(path)
ref_loader = DataLoader(ref_data, batch_size=batch_size, shuffle=True, num_workers=8)

if '128' in path:
	img_size = 128

# Model
feature_dim = args.feature_dim #if dataset == 'imagenet' else 50
net = Discriminator(img_size=img_size, feature_dim=feature_dim) if dataset == 'imagenet' else Discriminator_cifar(img_size=img_size, feature_dim=feature_dim)
net = net.cuda()

# Initialize parameters
# epsilonOPT = torch.log(torch.from_numpy(np.random.rand(1) * 10 ** (-10)).to(device, torch.float))
epsilonOPT = torch.log(torch.from_numpy(np.random.rand(1) * 10 ** (-args.epsilon_MMD)).to(device, torch.float))
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

def plot_mi(clean, adv, model=None):

	if clean.is_cuda or adv.is_cuda:
		clean = clean.cpu()
		adv = adv.cpu()
	mi_nat = clean.numpy().astype(np.float64)
	mi_svhn = adv.numpy().astype(np.float64)

	mi_nat = mi_nat[~np.isnan(mi_nat)]
	mi_svhn = mi_svhn[~np.isnan(mi_svhn)]

	x = np.concatenate((mi_nat, mi_svhn), 0)
	y = np.zeros(x.shape[0])
	y[mi_nat.shape[0]:] = 1
	if model is None:
		model = LogisticRegressionCV(n_jobs=-1,cv=5).fit(x.reshape(-1, 1), y)
	y_pred = model.predict(x.reshape(-1, 1))

	accuracy = accuracy_score(y, y_pred)

	ap = metrics.roc_auc_score(y, x)
	fpr, tpr, thresholds = metrics.roc_curve(y, x)
	accs = {th: tpr[np.argwhere(fpr <= th).max()] for th in [0.01, 0.05, 0.1]}
	print("auroc: {:.4f}; ".format(ap) + "; ".join(["TPR: {:.4f} @ FPR={:.4f}".format(v, k) for k, v in accs.items()]) + "  {}-{}".format(len(mi_nat), len(mi_svhn)))
	return ap, accuracy, model

def get_train_data_loader():
	path_adv = f'{datapath}/scores_adv_FGSM_L2_0.00392_5single_vector_norm{t}perb_image1000/'
	adv_data1 = DatasetNPY(path_adv)
	adv_data_loader1 = DataLoader(adv_data1, batch_size=batch_size, shuffle=True, num_workers=8)

	path_adv2 = f'{datapath}/scores_adv_FGSM_0.00392_5single_vector_norm{t}perb_image1000/'
	adv_data2 = DatasetNPY(path_adv2)
	adv_data_loader2 = DataLoader(adv_data2, batch_size=batch_size, shuffle=True, num_workers=8)

	return adv_data_loader1, adv_data_loader2

def logistic_regression_model_train(model_path):
	adv_data_loader1, adv_data_loader2 = get_train_data_loader()
	fea_reference_ls = []
	feature_cln_ls = []
	feature_adv_ls = []
	with torch.no_grad():
		for batch_idx, (inputs,x_adv1, x_adv2) in enumerate(zip(ref_loader, adv_data_loader1, adv_data_loader2)):
			# if len(fea_reference_ls)>SIZE:
			# 	break
			if inputs.shape[0]!=x_adv1.shape[0] or x_adv1.shape[0] != x_adv2.shape[0]:
				break
			# assert inputs.shape[0]==x_adv1.shape[0] and x_adv1.shape[0] == x_adv2.shape[0]
			x_clean = inputs[:inputs.shape[0]//2,:,:,:].cuda(non_blocking=True)
			x_ref = inputs[inputs.shape[0]//2:,:,:,:].cuda(non_blocking=True)
			x_adv1= x_adv1[:inputs.shape[0]//2,:,:,:].cuda(non_blocking=True)
			x_adv2= x_adv2[:inputs.shape[0]//2,:,:,:].cuda(non_blocking=True)
			x_adv = torch.cat([x_adv1,x_adv2],dim=0)
			_,feature_ref = net(x_ref,out_feature=True)
			_,feature_clean = net(x_clean,out_feature=True)
			_,feature_adv = net(x_adv,out_feature=True)

			fea_reference_ls.append(feature_ref)
			feature_cln_ls.append(feature_clean)
			feature_adv_ls.append(feature_adv)
		

		ref_data = torch.cat(fea_reference_ls,dim=0)[:SIZE].cuda()
		x_cln = torch.cat(feature_cln_ls,dim=0).cuda()
		x_adv = torch.cat(feature_adv_ls,dim=0).cuda()

		# calculate the MMD
		dt_clean = MMD_batch2(torch.cat([ref_data,x_cln],dim=0), ref_data.shape[0], 0, sigma, sigma0_u, ep,is_smooth=False).cpu()	
		dt_adv = MMD_batch2(torch.cat([ref_data,x_adv],dim=0), ref_data.shape[0], 0, sigma, sigma0_u, ep,is_smooth=False).cpu()
		auroc_value_train, accuracy_train, model = plot_mi(dt_clean, dt_adv)
		print('auroc_value_train:', auroc_value_train, 'accuracy_train:', accuracy_train)

	joblib.dump(model, model_path + '/'+ 'logistic_regression_model.pkl')
	torch.save(ref_data, model_path + '/'+ 'feature_ref_for_test.pth')

def EPS_fea_get(args, config, DEVICE, loader=None):
	# 1) load model and data
	from eval_epsad import  SDE_Adv_Model
	print('starting the model and loader...')
	model = SDE_Adv_Model(args, config)         
	model = model.eval().to(DEVICE)

	if loader is None:
		loader = load_detection_data(args, args.adv_batch_size)

	# get the score function
	from score_sde import sde_lib
	from score_sde.models import utils as mutils
	rev_vpsde = model.runner.rev_vpsde
	sde = sde_lib.VPSDE(beta_min=rev_vpsde.beta_0, beta_max=rev_vpsde.beta_1, N=rev_vpsde.N)
	score_fn = mutils.get_score_fn(sde, rev_vpsde.model, train=False, continuous=True)

	# 2) # test the model
	mean = torch.zeros(1, 3, 1, 1).to(DEVICE)
	std = torch.ones(1, 3, 1, 1).to(DEVICE)
	classifier = get_image_classifier(args.classifier_name).to(config.device)
	score_adv_lists = []
	for i, (x, y) in enumerate(loader):
		x = x.to(config.device)
		y = y.to(config.device)

		if not args.clean_score_flag:
			attack_method = args.attack_methods[0] # 'FGSM_L2', 'PGD_L2', 'BIM_L2', 'MM_Attack', 'AA_Attack'
			if attack_method == 'MM_Attack' or attack_method == 'AA_Attack':
				with torch.no_grad():
					output = classifier(x)
				correct_batch = y.eq(output.max(dim=1)[1])
				x = x[correct_batch]
				y = y[correct_batch]
			x_adv, top1, top5 = ens_attack(x, y, classifier, mean, std, args, attack_method)
		else:
			x_adv = x
			top1, top5 = 1, 1
		
		score_sum = torch.zeros_like(x_adv, device=x_adv.device)
		with torch.no_grad():
			for value in range(1,args.diffuse_t+1):
				# if value>3:
				# 	break
				t_valuve = value/1000
				curr_t_temp = torch.tensor(t_valuve,device=x.device)

				if args.perb_image:
					z = torch.randn_like(x_adv, device=x_adv.device)
					mean_x_adv, std_x_adv = sde.marginal_prob(2*x_adv-1, curr_t_temp.expand(x_adv.shape[0]))
					perturbed_data = mean_x_adv + std_x_adv[:, None, None, None] * z
					score = score_fn(perturbed_data, curr_t_temp.expand(x_adv.shape[0]))
				else:
					score = score_fn(2*x_adv-1, curr_t_temp.expand(x_adv.shape[0]))
				if args.domain=='imagenet':
					score, _ = torch.split(score, score.shape[1]//2, dim=1)
					assert x_adv.shape == score.shape, f'{x_adv.shape}, {score.shape}'
				score_sum += score.detach()
			score_tensor = score_sum/args.diffuse_t
			score_adv_lists.append(score_tensor)
	score_tensor = torch.cat(score_adv_lists, dim=0)
	print('score_tensor:', score_tensor.shape)

	return score_tensor

def feature_test_get(score_tensor, net, device, adv_batch_size=args.adv_batch_size):
	feature_test_ls = []
	for i in range(0, score_tensor.shape[0], adv_batch_size):
		score_tensor_batch = score_tensor[i:i+adv_batch_size]
		_,feature_test = net(score_tensor_batch,out_feature=True)
		feature_test_ls.append(feature_test)
	feature_test = torch.cat(feature_test_ls, dim=0)
	return feature_test

if True:
	epoch = 99
	print('==> testing from checkpoint..')
	model_path = f'./net_D/{args.dataset}/{id}'
	assert os.path.isdir(model_path), 'Error: no checkpoint directory found!'
	# checkpoint = torch.load(model_path + '/'+ str(epoch) +'last_ckpt.pth')
	checkpoint = torch.load(model_path + '/'+ 'last_ckpt.pth')
	net.load_state_dict(checkpoint['net'])
	net.eval()
	sigma, sigma0_u, ep  = checkpoint['sigma'], checkpoint['sigma0_u'], checkpoint['ep']
	if not os.path.exists(model_path + '/'+ 'logistic_regression_model.pkl') or not os.path.exists(model_path + '/'+ 'feature_ref_for_test.pth'):
		logistic_regression_model_train(model_path)
	load_ref_data =  torch.load(model_path + '/'+ 'feature_ref_for_test.pth') # cpu
	logistic_regression_model = joblib.load(model_path + '/'+ 'logistic_regression_model.pkl') # cpu
	
	sigma, sigma0_u, ep  = sigma.to(device), sigma0_u.to(device), ep.to(device)
	load_ref_data = load_ref_data.to(device)

	# predicting images from a directory
	loader=args.loader # This is your preferred data loader
	score_tensor = EPS_fea_get(args, config, device,loader)

	# test the model
	with torch.no_grad():

		feature_test = feature_test_get(score_tensor, net, device, args.adv_batch_size)
		dt_test = MMD_batch2(torch.cat([load_ref_data,feature_test],dim=0), load_ref_data.shape[0], 0, sigma, sigma0_u, ep,is_smooth=False).cpu()
	
	y_pred_loaded = logistic_regression_model.predict(dt_test.detach().numpy().reshape(-1, 1))
	# print("y_pred_loaded:", y_pred_loaded)
	if args.clean_score_flag:
		acc = accuracy_score(np.zeros(dt_test.shape[0]), y_pred_loaded)
		print('clean_acc:', acc)
	else:
		acc = accuracy_score(np.ones(dt_test.shape[0]), y_pred_loaded)
		print('adv_acc:', acc)	
