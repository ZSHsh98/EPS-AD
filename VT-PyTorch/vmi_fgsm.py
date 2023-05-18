# coding=utf-8
'''Implementation of MI-FGSM attack in PyTorch'''
import argparse
import os.path

from utils import *

import torch
import torch.nn as nn
import torchvision.models as fp_models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from torch.nn import functional as F


parser = argparse.ArgumentParser(description='VT-attack in PyTorch')
parser.add_argument('--batch_size', type=int, default=10, help='mini-batch size (default: 1)')  # no_samples 要能被 batch_size 整除
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--no_samples', default=1000, type=int, help='the number of samples selected from the whole dataset for test')
parser.add_argument('--max_epsilon', default=16.0, type=float, help='max magnitude of adversarial perturbations')
parser.add_argument('--num_iter', default=10, type=int, help='max iteration')
parser.add_argument('--momentum', default=1.0, type=float, help='momentum of the attack')
parser.add_argument('--output_dir', default='./data/adv/vmi-fgsm/inceptionv3', type=str, help='directory of crafted adversarial examples')
parser.add_argument('--clean_dir', default='./data/val_clean', type=str, help='directory for test')
parser.add_argument('--model_dir_fp', default='./checkpoints/fp_models', type=str, help='directory for load')
parser.add_argument('--victim_model', default='inceptionv3', type=str, help='directory for test')
parser.add_argument('--number', default=20, type=int, help='the number of images for variance tuning')
parser.add_argument('--beta', default=1.5, type=float, help='the bound for variance tuning')
parser.add_argument('--device', default='3', type=str, help='gpu device')
parser.add_argument('--use_gpu', default=True, type=bool, help='')

args = parser.parse_args()



def vmi_fgsm(x, y, model, use_gpu=True):
    '''
    craft adversarial examples
    :param x: clean images in batch in [-1, 1]
    :param y: correct labels
    :return: adv in [-1, 1]
    '''
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
        adv_x, corr_1, corr_5 = vmi_fgsm(x, y, model, use_gpu=args.use_gpu)
        correct_1 += corr_1
        correct_5 += corr_5
        save_images(adv_x, i, batch_size=args.batch_size, output_dir=args.output_dir)
        print('attack in process, i = %d, top1 = %.3f, top5 = %.3f' %
              (i, corr_1 / args.batch_size, corr_5 / args.batch_size))

    print('attack finished')
    print('memory image attack: top1 = %.3f, top5 = %.3f' %
          (correct_1 / args.no_samples, correct_5 / args.no_samples))




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()