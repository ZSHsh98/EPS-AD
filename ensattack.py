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
					# outputs, __ = model(X.sub(mean).div(std))[0:2]
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

	def input_diversity(args, input_tensor):
		'''apply input transformation to enhance transferability: padding and resizing (DIM)'''
		image_size = 224 if args.domain == 'imagenet' else 32
		rnd = torch.randint(image_size, args.image_resize, ())   # uniform distribution
		rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='nearest')
		h_rem = args.image_resize - rnd
		w_rem = args.image_resize - rnd
		pad_top = torch.randint(0, h_rem, ())
		pad_bottom = h_rem - pad_top
		pad_left = torch.randint(0, w_rem, ())
		pad_right = w_rem - pad_left
		# pad的参数顺序在pytorch里面是左右上下，在tensorflow里是上下左右，而且要注意pytorch的图像格式是BCHW, tensorflow是CHWB
		padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom, 0, 0, 0, 0))
		if torch.rand(1) < args.prob:
			ret = padded
		else:
			ret = input_tensor
		ret = F.interpolate(ret, [image_size, image_size], mode='nearest')
		return ret

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

	def _CW_L2_whitebox(X, y, mean, std):
		bs = X.shape[0]
		scale_ = np.sqrt(np.prod(list(X.shape[1:])))
		lr = args.step_size_adv * scale_
		radius = args.epsilon * scale_

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

			grad_norm_ = torch.clamp(torch.norm(X_cw.grad.view(bs, -1), dim=1), min=1e-12).view(bs, 1, 1, 1)
			grad_unit_ = X_cw.grad / grad_norm_
			X_cw += lr * grad_unit_
			eta = X_cw - X
			eta_norm = torch.clamp(torch.norm(eta.view(bs, -1), dim=1), min=radius).view(bs, 1, 1, 1)
			eta = eta * (radius / eta_norm)
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
		adversary_resnet = AutoAttack(model, norm=args.lp_norm, eps=args.epsilon,
								  version=attack_version)
		adversary_resnet.apgd.n_iter = args.num_steps
		adversary_resnet.fab.n_iter = args.num_steps
		adversary_resnet.apgd_targeted.n_iter = args.num_steps
		
		x_adv = adversary_resnet.run_standard_evaluation(X, y, bs=X.shape[0])

		return x_adv

	
	def _MM_Attack_whitebox(model, X, y, k=3):
		import autoattack
		from MMattack.attack_apgd import APGDAttack, APGDAttack_targeted
		# apgd = APGDAttack_targeted(model, n_restarts=1, n_iter=perturb_steps, verbose=False,
        #         eps=args.epsilon, norm='Linf', eot_iter=1, rho=.75, seed=1, device='cuda')
		apgd = APGDAttack_targeted(model, n_restarts=1, n_iter=args.num_steps, verbose=False,
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

	def _VMI_FGSM_whitebox(model, X, y):
		model = model.cuda()

		x = X * 2 - 1

		num_iter = args.num_steps
		eps = args.epsilon * 2.0
		alpha = eps / num_iter
		momentum = args.momentum
		number = args.number
		beta = args.beta
		grads = torch.zeros_like(x,requires_grad=False)
		variance = torch.zeros_like(x,requires_grad=False)
		min_x = x - eps
		max_x = x + eps

		adv = x.clone()
		
		with torch.enable_grad():
			for i in range(num_iter):
				adv.requires_grad = True
				outputs = model(adv)
				loss = F.cross_entropy(outputs, y)
				loss.backward()
				new_grad = adv.grad
				noise = momentum * grads + (new_grad + variance) / torch.norm(new_grad + variance, p=1)

				# update variance
				sample = adv.clone().detach()
				global_grad = torch.zeros_like(x, requires_grad=False)
				for _ in range(number):
					sample = sample.detach()
					sample.requires_grad = True
					rd = (torch.rand_like(x) * 2 - 1) * beta * eps
					sample = sample + rd
					outputs_sample = model(sample)
					loss_sample = F.cross_entropy(outputs_sample, y)
					global_grad += torch.autograd.grad(loss_sample, sample, grad_outputs=None, only_inputs=True)[0]
				variance = global_grad / (number * 1.0) - new_grad

				adv = adv + alpha * noise.sign()
				adv = torch.clamp(adv, -1.0, 1.0).detach()   # range [-1, 1]
				adv = torch.max(torch.min(adv, max_x), min_x).detach()
				grads = noise
		
		return (adv + 1) / 2
	def _VMI_CT_FGSM_whitebox(model, X, y):
		model = model.cuda()

		x = X * 2 - 1

		num_iter = args.num_steps
		eps = args.epsilon * 2.0
		alpha = eps / num_iter
		momentum = args.momentum
		number = args.number
		beta = args.beta
		grads = torch.zeros_like(x,requires_grad=False)
		variance = torch.zeros_like(x,requires_grad=False)
		min_x = x - eps
		max_x = x + eps

		adv = x.clone()
		y_batch = torch.cat((y, y, y, y, y),dim=0)

		with torch.enable_grad():
			for i in range(num_iter):
				adv.requires_grad = True
				x_batch = torch.cat((adv, adv/2., adv/4., adv/8., adv/16.),dim=0)
				outputs = model(input_diversity(args, x_batch))
				loss = F.cross_entropy(outputs, y_batch)
				grad_vanilla = torch.autograd.grad(loss, x_batch, grad_outputs=None, only_inputs=True)[0]
				grad_batch_split = torch.split(grad_vanilla, split_size_or_sections=args.adv_batch_size, dim=0)
				grad_in_batch = torch.stack(grad_batch_split, dim=4)
				new_grad = torch.sum(grad_in_batch * torch.tensor([1., 1 / 2., 1 / 4., 1 / 8, 1 / 16.]).cuda(), dim=4, keepdim=False)
				current_grad = new_grad + variance
				noise = F.conv2d(input=current_grad, weight=stack_kernel, stride=1, padding=3, groups=3)
				noise = momentum * grads + noise / torch.norm(noise, p=1)

				# update variance
				sample = x_batch.clone().detach()
				global_grad = torch.zeros_like(x, requires_grad=False)
				for _ in range(number):
					sample = sample.detach()
					sample.requires_grad = True
					rd = (torch.rand_like(x) * 2 - 1) * beta * eps
					rd_batch = torch.cat((rd, rd / 2., rd / 4., rd / 8., rd / 16.), dim=0)
					sample = sample + rd_batch
					outputs_sample = model(input_diversity(args, sample))
					loss_sample = F.cross_entropy(outputs_sample, y_batch)
					grad_vanilla_sample = torch.autograd.grad(loss_sample, sample, grad_outputs=None, only_inputs=True)[0]
					grad_batch_split_sample = torch.split(grad_vanilla_sample, split_size_or_sections=args.adv_batch_size,
														dim=0)
					grad_in_batch_sample = torch.stack(grad_batch_split_sample, dim=4)
					global_grad += torch.sum(grad_in_batch_sample * torch.tensor([1., 1 / 2., 1 / 4., 1 / 8, 1 / 16.]).cuda(), dim=4, keepdim=False)
				variance = global_grad / (number * 1.0) - new_grad

				adv = adv + alpha * noise.sign()
				adv = torch.clamp(adv, -1.0, 1.0).detach()   # range [-1, 1]
				adv = torch.max(torch.min(adv, max_x), min_x).detach()
				grads = noise
		
		return (adv + 1) / 2



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

			if attack_method in ['MM_Attack','VMI_FGSM']:
				X_adv = eval('_{}_whitebox'.format(attack_method))(model, input, target)
			elif attack_method in ['AA_Attack']:
				X_adv = eval('_{}_whitebox'.format(attack_method))(model, input, target, args)
			else:
				X_adv = eval('_{}_whitebox'.format(attack_method))(input, target, mean, std)

			outputs = model(X_adv.sub(mean).div(std))
			# outputs, __ = model(X_adv.sub(mean).div(std))[0:2]
			prec1, prec5 = accuracy(outputs, target, topk=(1, 5))

			# losses += loss * target.size(0)
			top1 += prec1.item() * target.size(0)/100
			top5 += prec5.item() * target.size(0)/100
			num_data += target.size(0)

		# losses /= num_data
		top1 /= num_data
		top5 /= num_data

	return X_adv, top1, top5