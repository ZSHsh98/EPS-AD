# coding=utf-8
'''Implementation of MI-FGSM attack in PyTorch'''
import argparse
import numpy as np
import os.path
import scipy.stats as st

from utils import *

import torch
import torch.nn as nn
import torchvision.models as fp_models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from torch.nn import functional as F


parser = argparse.ArgumentParser(description='VT-attack in PyTorch')
parser.add_argument('--batch_size', type=int, default=5, help='mini-batch size (default: 10)')  # no_samples 要能被 batch_size 整除
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--no_samples', default=1000, type=int, help='the number of samples selected from the whole dataset for test')
parser.add_argument('--max_epsilon', default=16.0, type=float, help='max magnitude of adversarial perturbations')
parser.add_argument('--num_iter', default=10, type=int, help='max iteration')
parser.add_argument('--momentum', default=1.0, type=float, help='momentum of the attack')
parser.add_argument('--output_dir', default='./data/adv/vmi-di-ti-si-fgsm/inceptionv3', type=str, help='directory of crafted adversarial examples')
parser.add_argument('--clean_dir', default='./data/val_clean', type=str, help='directory for test')
parser.add_argument('--model_dir_q', default='./checkpoints/q_models', type=str, help='root directory for model')
parser.add_argument('--victim_model', default='inceptionv3', type=str, help='victim model')
parser.add_argument('--number', default=20, type=int, help='the number of images for variance tuning')
parser.add_argument('--beta', default=1.5, type=float, help='the bound for variance tuning')
parser.add_argument('--device', default='1', type=str, help='gpu device')
parser.add_argument('--image_height', default=299, type=int, help='height of each input image')
parser.add_argument('--image_width', default=299, type=int, help='width of each input image')
parser.add_argument('--image_resize', default=331, type=int, help='heigth of each input image')
parser.add_argument('--prob', default=0.5, type=float, help='probability of using diverse inputs')
parser.add_argument('--use_gpu', default=True, type=bool, help='')

args = parser.parse_args()

def input_diversity(input_tensor):
    '''apply input transformation to enhance transferability: padding and resizing (DIM)'''
    rnd = torch.randint(args.image_width, args.image_resize, ())   # uniform distribution
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
    ret = F.interpolate(ret, [args.image_height, args.image_width], mode='nearest')
    return ret


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel



def vmi_di_ti_si_fgsm(x, y, model, use_gpu=True):
    '''
    craft adversarial examples
    :param x: clean images in batch in [-1, 1]
    :param y: correct labels
    :return: adv in [-1, 1]
    '''
    kernel = gkern(7, 3).astype(np.float32)
    # 要注意Pytorch是BCHW, tensorflow是BHWC
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)  # batch, channel, height, width = 3, 1, 7, 7
    stack_kernel = torch.tensor(stack_kernel).cuda()


    model = model.eval()
    if use_gpu:
        x = x.cuda()
        y = y.cuda()
        model = model.cuda()

    x = x * 2 - 1

    num_iter = args.num_iter
    eps = args.max_epsilon / 255 * 2.0
    alpha = eps / num_iter   # attack step size
    momentum = args.momentum
    number = args.number
    beta = args.beta
    grads = torch.zeros_like(x, requires_grad=False)
    variance = torch.zeros_like(x, requires_grad=False)
    min_x = x - eps
    max_x = x + eps

    adv = x.clone()
    y_batch = torch.cat((y, y, y, y, y), dim=0)

    with torch.enable_grad():
        for i in range(num_iter):
            adv.requires_grad = True
            x_batch = torch.cat((adv, adv / 2., adv / 4., adv / 8., adv / 16.), dim=0)
            outputs = model(input_diversity(x_batch))
            loss = F.cross_entropy(outputs, y_batch)
            grad_vanilla = torch.autograd.grad(loss, x_batch, grad_outputs=None, only_inputs=True)[0]
            grad_batch_split = torch.split(grad_vanilla, split_size_or_sections=args.batch_size, dim=0)
            grad_in_batch = torch.stack(grad_batch_split, dim=4)
            if use_gpu:
                new_grad = torch.sum(grad_in_batch * torch.tensor([1., 1 / 2., 1 / 4., 1 / 8, 1 / 16.]).cuda(), dim=4, keepdim=False)
            else:
                new_grad = torch.sum(grad_in_batch * torch.tensor([1., 1 / 2., 1 / 4., 1 / 8, 1 / 16.]), dim=4, keepdim=False)
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
                outputs_sample = model(input_diversity(sample))
                loss_sample = F.cross_entropy(outputs_sample, y_batch)
                grad_vanilla_sample = torch.autograd.grad(loss_sample, sample, grad_outputs=None, only_inputs=True)[0]
                grad_batch_split_sample = torch.split(grad_vanilla_sample, split_size_or_sections=args.batch_size,
                                                      dim=0)
                grad_in_batch_sample = torch.stack(grad_batch_split_sample, dim=4)
                if use_gpu:
                    global_grad += torch.sum(grad_in_batch_sample * torch.tensor([1., 1 / 2., 1 / 4., 1 / 8, 1 / 16.]).cuda(), dim=4, keepdim=False)
                else:
                    global_grad += torch.sum(grad_in_batch_sample * torch.tensor([1., 1 / 2., 1 / 4., 1 / 8, 1 / 16.]), dim=4, keepdim=False)
            variance = global_grad / (number * 1.0) - new_grad

            adv = adv + alpha * noise.sign()
            adv = torch.clamp(adv, -1.0, 1.0).detach()   # range [-1, 1]
            adv = torch.max(torch.min(adv, max_x), min_x).detach()
            grads = noise

    '''validate in memory'''
    output = model(adv)
    pred_top1 = output.topk(k=1, largest=True).indices
    pred_top5 = output.topk(k=5, largest=True).indices
    if pred_top1.dim() >= 2:
        pred_top1 = pred_top1.squeeze()


    return (adv + 1) / 2, (pred_top1 == y).sum().item(), \
           (pred_top5 == y.unsqueeze(dim=1).expand(-1, 5)).sum().item()



def main():

    model_name = args.victim_model
    model = load_model(name=model_name).cuda()


    # prepare dataset
    if model_name == 'inceptionv3':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])


    clean_dataset = DataSet(img_dir=args.clean_dir, transform=preprocess, png=False)

    clean_loader = torch.utils.data.DataLoader(
        clean_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    correct_1 = 0
    correct_5 = 0
    for i, (x, y) in enumerate(clean_loader):
        if args.use_gpu:
            x = x.cuda()
            y = y.cuda()
        adv_x, corr_1, corr_5 = vmi_di_ti_si_fgsm(x, y, model, use_gpu=args.use_gpu)
        correct_1 += corr_1
        correct_5 += corr_5
        save_images(adv_x, i, batch_size=args.batch_size, output_dir=args.output_dir, png=False)
        print('attack in process, i = %d, top1 = %.3f, top5 = %.3f' %
              (i, corr_1 / args.batch_size, corr_5 / args.batch_size))

    print('attack finished')
    print('memory image attack: top1 = %.3f, top5 = %.3f' %
          (correct_1 / args.no_samples, correct_5 / args.no_samples))




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()
