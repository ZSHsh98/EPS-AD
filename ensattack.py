from tkinter import Y
import torch
import numpy as np
import torch.nn.functional as F
import os, random
import torch.distributed as dist
def get_rank():
	if not dist.is_available():
		return 0

	if not dist.is_initialized():
		return 0

	return dist.get_rank()
def gaussian_kernel():
	def gkern(kernlen=21, nsig=3):
		"""Returns a 2D Gaussian kernel array."""
		import scipy.stats as st

		x = np.linspace(-nsig, nsig, kernlen)
		kern1d = st.norm.pdf(x)
		kernel_raw = np.outer(kern1d, kern1d)
		kernel = kernel_raw / kernel_raw.sum()
		return kernel
	kernel = gkern(7, 3).astype(np.float32)
	stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
	stack_kernel = np.expand_dims(stack_kernel, 3)
	stack_kernel = stack_kernel.transpose((2, 3, 0, 1))
	stack_kernel = torch.from_numpy(stack_kernel)
	return stack_kernel
	
def smooth(x, stack_kernel):
	''' implemenet depthwiseConv with padding_mode='SAME' in pytorch '''
	padding = (stack_kernel.size(-1) - 1) // 2
	groups = x.size(1)
	return torch.nn.functional.conv2d(x, weight=stack_kernel, padding=padding, groups=groups)

def accuracy(output, target, topk=(1,)):
	if len(target.shape) > 1: return torch.tensor(1), torch.tensor(1)

	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t().contiguous()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
	return res

def mm_loss(output, target, target_choose, confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = torch.autograd.Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)

    target_onehot = torch.zeros(target_choose.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target_choose.unsqueeze(1), 1.)
    target_var = torch.autograd.Variable(target_onehot, requires_grad=False)

    other = (target_var * output).sum(1)
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

def ens_attack(input, target, model, mean, std, args, attack_method, attack_model=None):
	def _grad(X, y, mean, std):
		with torch.enable_grad():
					X.requires_grad_()
					outputs = attack_model(X.sub(mean).div(std))
					outputs = outputs.softmax(-1)
					if outputs.dim() == 3:
						output = outputs.mean(-2) + 1e-10
					else:
						output = outputs
					loss = F.cross_entropy(output.log(), y, reduction='none')
					# if not args.MM_attack_flag:
					# 	loss = F.cross_entropy(output.log(), y, reduction='none')
					# else:
					# 	loss = mm_loss(output, y, )

					# if args.mimicry != 0.:
					# 	loss -= features.var(dim=1).mean(dim=[1,2,3]) * args.mimicry
					grad_ = torch.autograd.grad(
						[loss], [X], grad_outputs=torch.ones_like(loss),
						retain_graph=False)[0].detach()
		return grad_

	def _PGD_whitebox(X, y, mean, std):
		X_pgd = X.clone()
		if args.random:
			X_pgd += torch.cuda.FloatTensor(*X_pgd.shape).uniform_(-args.epsilon, args.epsilon)
		for _ in range(args.num_steps):
			grad_ = _grad(X_pgd, y, mean, std)
			X_pgd += args.step_size_adv * grad_.sign()
			eta = torch.clamp(X_pgd - X, -args.epsilon, args.epsilon)
			X_pgd = torch.clamp(X + eta, 0, 1.0)
		return X_pgd

	def _PGD_L2_whitebox(X, y, mean, std):
		bs = X.shape[0]
		scale_ = np.sqrt(np.prod(list(X.shape[1:])))
		lr = args.step_size_adv * scale_
		radius = args.epsilon * scale_

		X_pgd = X.clone()
		if args.random:
			X_pgd += torch.cuda.FloatTensor(*X_pgd.shape).uniform_(-args.epsilon, args.epsilon)
		for _ in range(args.num_steps):
			grad_ = _grad(X_pgd, y, mean, std)
			grad_norm_ = torch.clamp(torch.norm(grad_.view(bs, -1), dim=1), min=1e-12).view(bs, 1, 1, 1)
			grad_unit_ = grad_ / grad_norm_
			X_pgd += lr * grad_unit_

			eta = X_pgd - X
			eta_norm = torch.clamp(torch.norm(eta.view(bs, -1), dim=1), min=radius).view(bs, 1, 1, 1)
			eta = eta * (radius / eta_norm)
			X_pgd = torch.clamp(X + eta, 0, 1.0)
		return X_pgd

	def _FGSM_whitebox(X, y, mean, std):
		X_fgsm = X.clone()
		grad_ = _grad(X_fgsm, y, mean, std)
		eta = args.epsilon * grad_.sign()
		X_fgsm = torch.clamp(X_fgsm + eta, 0, 1.0)
		return X_fgsm

	def _FGSM_RS_whitebox(X, y, mean, std):
		X_fgsm = X.clone()
		X_fgsm += torch.cuda.FloatTensor(*X_fgsm.shape).uniform_(-args.epsilon, args.epsilon)
		grad_ = _grad(X_fgsm, y, mean, std)
		eta = args.epsilon * grad_.sign()
		X_fgsm = torch.clamp(X_fgsm + eta, 0, 1.0)
		return X_fgsm

	def _FGSM_L2_whitebox(X, y, mean, std):
		X_fgsm = X.clone()
		grad_ = _grad(X_fgsm, y, mean, std)
		grad_norm_ = torch.clamp(torch.norm(grad_.view(X.shape[0], -1), dim=1), min=1e-12).view(X.shape[0], 1, 1, 1)
		grad_unit_ = grad_ / grad_norm_
		eta = args.epsilon * np.sqrt(np.prod(list(X.shape[1:]))) * grad_unit_
		X_fgsm = torch.clamp(X_fgsm + eta, 0, 1.0)
		return X_fgsm

	def _BIM_whitebox(X, y, mean, std):
		X_bim = X.clone()
		for _ in range(args.num_steps):
			grad_ = _grad(X_bim, y, mean, std)
			X_bim += args.step_size_adv * grad_.sign()
			eta = torch.clamp(X_bim - X, -args.epsilon, args.epsilon)
			X_bim = torch.clamp(X + eta, 0, 1.0)
		return X_bim

	def _BIM_L2_whitebox(X, y, mean, std):
		bs = X.shape[0]
		scale_ = np.sqrt(np.prod(list(X.shape[1:])))
		lr = args.step_size_adv * scale_
		radius = args.epsilon * scale_

		X_bim = X.clone()
		for _ in range(args.num_steps):
			grad_ = _grad(X_bim, y, mean, std)
			grad_norm_ = torch.clamp(torch.norm(grad_.view(bs, -1), dim=1), min=1e-12).view(bs, 1, 1, 1)
			grad_unit_ = grad_ / grad_norm_
			X_bim += lr * grad_unit_

			eta = X_bim - X
			eta_norm = torch.clamp(torch.norm(eta.view(bs, -1), dim=1), min=radius).view(bs, 1, 1, 1)
			eta = eta * (radius / eta_norm)
			X_bim = torch.clamp(X + eta, 0, 1.0)
		return X_bim

	def _MIM_whitebox(X, y, mean, std):
		X_mim = X.clone()
		g = torch.zeros_like(X_mim)
		for _ in range(args.num_steps):
			grad_ = _grad(X_mim, y, mean, std)
			grad_ /= grad_.abs().mean(dim=[1,2,3], keepdim=True)
			g = g * args.mim_momentum + grad_
			X_mim += args.step_size_adv * g.sign()
			eta = torch.clamp(X_mim - X, -args.epsilon, args.epsilon)
			X_mim = torch.clamp(X + eta, 0, 1.0)
		return X_mim

	def _TIM_whitebox(X, y, mean, std):
		X_tim = X.clone()
		g = torch.zeros_like(X_tim)
		for _ in range(args.num_steps):
			grad_ = _grad(X_tim, y, mean, std)
			grad_ = smooth(grad_, stack_kernel)
			grad_ /= grad_.abs().mean(dim=[1,2,3], keepdim=True)
			g = g * args.mim_momentum + grad_
			X_tim += args.step_size_adv * g.sign()
			eta = torch.clamp(X_tim - X, -args.epsilon, args.epsilon)
			X_tim = torch.clamp(X + eta, 0, 1.0)
		return X_tim

	# def _MIM_L2_whitebox(X, y, mean, std):
	#     bs = X.shape[0]
	#     scale_ = np.sqrt(np.prod(list(X.shape[1:])))
	#     lr = args.step_size_adv * scale_
	#     radius = args.epsilon * scale_
	#
	#     X_mim = X.clone()
	#     g = torch.zeros_like(X_mim)
	#     for _ in range(args.num_steps):
	#         grad_ = _grad(X_mim, y, mean, std)
	#
	#     return X_mim

	def _CW_whitebox(X, y, mean, std):
		X_cw = X.clone()
		X_cw += torch.cuda.FloatTensor(*X_cw.shape).uniform_(-args.epsilon, args.epsilon)
		y_one_hot = F.one_hot(y, num_classes=args.num_classes)
		for _ in range(args.num_steps):
			X_cw.requires_grad_()
			if X_cw.grad is not None: del X_cw.grad
			X_cw.grad = None
			with torch.enable_grad():
				outputs = attack_model(X_cw.sub(mean).div(std))
				outputs = outputs.softmax(-1)
				if outputs.dim() == 3:
					logits = (outputs.mean(-2) + 1e-10).log()
				else:
					logits = outputs.log()
				logit_target = torch.max(y_one_hot * logits, 1)[0]
				logit_other = torch.max(
					(1 - y_one_hot) * logits - 1e6 * y_one_hot, 1)[0]
				loss = torch.mean(logit_other - logit_target)
				# if args.mimicry != 0.:
				# 	loss -= features.var(dim=1).mean() * args.mimicry
				loss.backward()

			X_cw += args.step_size_adv * X_cw.grad.sign()
			eta = torch.clamp(X_cw - X, -args.epsilon, args.epsilon)
			X_cw = torch.clamp(X + eta, 0, 1.0)
		return X_cw

	def _DI_MIM_whitebox(X, y, mean, std):
		def Resize_and_padding(x, scale_factor=1.1):
			ori_size = x.size(-1)
			new_size = int(x.size(-1) * scale_factor)
			delta_w = new_size - ori_size
			delta_h = new_size - ori_size
			top = random.randint(0, delta_h)
			left = random.randint(0, delta_w)
			bottom = delta_h - top
			right = delta_w - left
			x = F.pad(x, pad=(left,right,top,bottom), value=0)
			return F.interpolate(x, size = ori_size)

		X_mim = X.clone()
		g = torch.zeros_like(X_mim)
		for _ in range(args.num_steps):
			grad_ = _grad(Resize_and_padding(X_mim), y, mean, std)
			grad_ /= grad_.abs().mean(dim=[1,2,3], keepdim=True)
			g = g * args.mim_momentum + grad_
			X_mim += args.step_size_adv * g.sign()
			eta = torch.clamp(X_mim - X, -args.epsilon, args.epsilon)
			X_mim = torch.clamp(X + eta, 0, 1.0)
		return X_mim
	
	def _AA_Attack_whitebox(model, X, y, args):
		from autoattack import AutoAttack

		attack_version = args.attack_version  # ['standard', 'rand', 'custom']
		if attack_version == 'standard':
			attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
		elif attack_version == 'rand':
			attack_list = ['apgd-ce', 'apgd-dlr']
		elif attack_version == 'custom':
			attack_list = args.attack_type.split(',')
		else:
			raise NotImplementedError(f'Unknown attack version: {attack_version}!')
		adversary_resnet = AutoAttack(model, norm=args.lp_norm, eps=args.epsilon,
								  version=attack_version, attacks_to_run=attack_list)
		adversary_resnet.apgd.n_iter = args.diffuse_t
		adversary_resnet.fab.n_iter = args.diffuse_t
		adversary_resnet.square.n_iter = args.diffuse_t
		adversary_resnet.apgd_targeted.n_iter = args.diffuse_t
		
		x_adv = adversary_resnet.run_standard_evaluation(X, y, bs=X.shape[0])

		return x_adv

	
	def _MM_Attack_whitebox(model, X, y, k=3):
		import autoattack
		from MMattack.attack_apgd import APGDAttack, APGDAttack_targeted
		# apgd = APGDAttack_targeted(model, n_restarts=1, n_iter=perturb_steps, verbose=False,
        #         eps=args.epsilon, norm='Linf', eot_iter=1, rho=.75, seed=1, device='cuda')
		apgd = APGDAttack_targeted(model, n_restarts=1, n_iter=args.diffuse_t, verbose=False,
                eps=args.epsilon, norm='Linf', eot_iter=1, rho=.75, seed=1, device='cuda')

		with torch.no_grad():
			
			robust_flags = torch.zeros(len(X),dtype=torch.bool)
			
			x0 = X.clone().cuda()
			y0 = y.clone().cuda()

			output = model(x0)
            
			correct_batch = y0.eq(output.max(dim=1)[1])
			robust_flags = correct_batch.detach()
		
		for i in range(k):
			num_robust = torch.sum(robust_flags).item()
			robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
			
			if num_robust >= 1:
				robust_lin_idcs.squeeze_()
			else:
				break

			xk = x0[robust_lin_idcs].clone().cuda()
			yk = y0[robust_lin_idcs].clone().cuda()

			apgd.n_target_classes = 1
			apgd.loss = 'mm'
			x_adv0 = apgd.perturb(xk, yk, i=i)

			x0[robust_lin_idcs] = x_adv0

			output = model(x_adv0)
			false_batch = ~yk.eq(output.max(dim=1)[1]).cuda()
			if num_robust == 1:
				non_robust_lin_idcs = robust_lin_idcs.unsqueeze(0)[false_batch]
			else:
				non_robust_lin_idcs = robust_lin_idcs[false_batch]
			robust_flags[non_robust_lin_idcs] = False

		return x0


	stack_kernel = gaussian_kernel().cuda()
	is_transferred = True if (attack_model is not None and attack_model != model) else False
	model.eval()
	# if not args.full:
	# 	parallel_eval(model)
	if is_transferred:
		attack_model.eval()
		# if not args.full:
		# 	parallel_eval(attack_model)
	else:
		attack_model = model

	with torch.no_grad():
		losses, top1, top5, num_data = 0, 0, 0, 0
		mis = []
		if 1:
		# for i, (input, target) in enumerate(val_loader):
			# if i>2:
			# 	break
			input = input.cuda(non_blocking=True).mul_(std).add_(mean)
			target = target.cuda(non_blocking=True)

			if attack_method == 'MM_Attack':
				X_adv = eval('_{}_whitebox'.format(attack_method))(model, input, target)
			elif attack_method in ['AA_Attack','KD']:
				X_adv = eval('_{}_whitebox'.format(attack_method))(model, input, target, args)
			else:
				X_adv = eval('_{}_whitebox'.format(attack_method))(input, target, mean, std)

			outputs = model(X_adv.sub(mean).div(std))
			prec1, prec5 = accuracy(outputs, target, topk=(1, 5))

			# losses += loss * target.size(0)
			top1 += prec1.item() * target.size(0)/100
			top5 += prec5.item() * target.size(0)/100
			num_data += target.size(0)

		# losses /= num_data
		top1 /= num_data
		top5 /= num_data

	return X_adv, top1, top5
