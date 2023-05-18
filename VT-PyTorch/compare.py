# coding=utf-8
'''Evaluate the attack success rate under quantized models'''
import argparse
import copy
import os.path
import pandas as pd

from utils import DataSet
from utils import validate

import torch
import torch.nn as nn

import torchvision.models.quantization as q_models
import torchvision.transforms as transforms
import torchvision.utils

from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader
from PIL import Image

parser = argparse.ArgumentParser(description='VT-attack in PyTorch')
parser.add_argument('--batch_size', type=int, default=100, help='mini-batch size')  # no_samples 要能被 batch_size 整除
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--no_samples', default=1000, type=int, help='the size of adv dataset')
parser.add_argument('--val_dir', default='./data/validation', type=str, help='directory for load')
parser.add_argument('--clean_dir', default='./data/val_clean', type=str, help='directory for load')
parser.add_argument('--adv_dir', default='./data/adv/vmi-di-ti-si-fgsm/inceptionv3', type=str, help='directory for load')
parser.add_argument('--official_adv_dir', default='./data/adv/official-vmi-di-ti-si-fgsm', type=str, help='directory for load')
parser.add_argument('--model_dir_q', default='./checkpoints/q_models', type=str, help='directory for load')
parser.add_argument('--model_dir_fp', default='./checkpoints/fp_models', type=str, help='directory for load')
parser.add_argument('--model', default='inceptionv3', type=str, help='model name')
parser.add_argument('--device', default='1', type=str, help='gpu id')


args = parser.parse_args()


def dynamic_quantization(model_fp):
    '''post training dynamic weight-only quantization. for fully connected layers, LSTM etc'''
    model_to_quantize = copy.deepcopy(model_fp)
    model_to_quantize.eval()
    model_quantized = torch.quantization.quantize_dynamic(model_to_quantize, qconfig_spec=None,
                                                          dtype=torch.qint8, mapping=None, inplace=False)
    torch.save(torch.jit.script(model_quantized), './static_q_models/inception_v3_google_q.pth')
    return model_quantized

def static_quantization(model_fp, val_loader):
    '''post training static quantization for both weights and activations. suitable for CNN'''

    '''model fusion for alexnet'''
    # torch.quantization.fuse_modules(model_fp, [['features.0', 'features.1'],
    #                                            ['features.3', 'features.4'],
    #                                            ['features.6', 'features.7'],
    #                                            ['features.8', 'features.9'],
    #                                            ['features.10', 'features.11'],
    #                                            ['classifier.1', 'classifier.2'],
    #                                            ['classifier.4', 'classifier.5']], inplace=True)

    model_fp = nn.Sequential(torch.quantization.QuantStub(), model_fp, torch.quantization.DeQuantStub())
    model_prepared = copy.deepcopy(model_fp)
    model_prepared.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model_prepared, inplace=True)

    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            output = model_prepared(input)
            if i % 10 == 0:
                print('feeding the data, batch = ', i)
            if i > 200:
                break

    model_quantized = torch.quantization.convert(model_prepared, inplace=True)
    torch.save(torch.jit.script(model_quantized), './static_q_models/inception_v3_google_q.pth')

model_map_quantized = {
    # imagenet pre-trained models
    'googlenet': 'googlenet_fbgemm-c00238cf.pth',
    'inceptionv3': 'inception_v3_google_fbgemm-71447a44.pth',
    'mobilenetv2': 'mobilenet_v2_qnnpack_37f702c5.pth',
    'resnet18': 'resnet18_fbgemm_16fa66dd.pth',
    'resnet50': 'resnet50_fbgemm_bf931d71.pth',
    'resnext101': 'resnext101_32x8_fbgemm_09835ccf.pth',
    'shufflenetv2': 'shufflenetv2_x1_fbgemm-db332c57.pth',
    'resnet34-2bit': 'res34_2bit_best.pth.tar',
    'resnet34-3bit': 'res34_3bit_best.pth.tar',
    'resnet34-4bit': 'res34_4bit_best.pth.tar',
    'resnet34-5bit': 'res34_5bit_best.pth.tar',
    'resnet18-2bit': 'res18_2bit_best.pth.tar',
}

model_map_fp = {
    'googlenet': 'googlenet-1378be20.pth',
    'inceptionv3': 'inception_v3_google-0cc3c7bd.pth',
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnext101': 'resnext101_32x8d-8ba56ff5.pth',
    'shufflenetv2': 'shufflenetv2_x1-5666bf0f80.pth',
    'mobilenetv2': 'mobilenet_v2-b0353104.pth'
}

def load_model(name, quantized):

    if quantized:
        model_map = model_map_quantized
        model_dir = os.path.join(args.model_dir_q, model_map[name])
    else:
        model_map = model_map_fp
        model_dir = os.path.join(args.model_dir_fp, model_map[name])

    if name == 'resnet18':
        model = q_models.resnet18(pretrained=False, quantize=quantized)
        model.load_state_dict(torch.load(model_dir))
    elif name == 'resnet50':
        model = q_models.resnet50(pretrained=False, quantize=quantized)
        model.load_state_dict(torch.load(model_dir))
    elif name == 'googlenet':
        model = q_models.googlenet(pretrained=False, quantize=quantized)
        model.load_state_dict(torch.load(model_dir))
    elif name == 'inceptionv3':
        model = q_models.inception_v3(pretrained=True, quantize=quantized)
        # model.load_state_dict(torch.load(model_dir))
    elif name == 'resnext101':
        model = q_models.resnext101_32x8d(pretrained=False, quantize=quantized)
        model.load_state_dict(torch.load(model_dir))
    elif name == 'shufflenetv2':
        model = q_models.shufflenet_v2_x1_0(pretrained=False, quantize=quantized)
        model.load_state_dict(torch.load(model_dir))
    elif name == 'mobilenetv2':
        model = q_models.mobilenet_v2(pretrained=False, quantize=quantized)
        model.load_state_dict(torch.load(model_dir))
    elif name == 'resnet18-2bit':
        print("loading checkpoint...")
        model = apot_models.__dict__['resnet18'](pretrained=True, bit=2)
        model = nn.DataParallel(model).cuda()
        checkpoint = torch.load(model_dir)
        model.load_state_dict(checkpoint['model'])
    elif name == 'resnet34-2bit':
        print("loading checkpoint...")
        model = apot_models.__dict__['resnet34'](pretrained=True, bit=2)
        model = nn.DataParallel(model).cuda()
        checkpoint = torch.load(model_dir)
        model.load_state_dict(checkpoint['model'])
    elif name == 'resnet34-3bit':
        print("loading checkpoint...")
        model = apot_models.__dict__['resnet34'](pretrained=True, bit=3)
        model = nn.DataParallel(model).cuda()
        checkpoint = torch.load(model_dir)
        model.load_state_dict(checkpoint['model'])
    elif name == 'resnet34-4bit':
        print("loading checkpoint...")
        model = apot_models.__dict__['resnet34'](pretrained=True, bit=4)
        model = nn.DataParallel(model).cuda()
        checkpoint = torch.load(model_dir)
        model.load_state_dict(checkpoint['model'])
    elif name == 'resnet34-5bit':
        print("loading checkpoint...")
        model = apot_models.__dict__['resnet34'](pretrained=True, bit=5)
        model = nn.DataParallel(model).cuda()
        checkpoint = torch.load(model_dir)
        model.load_state_dict(checkpoint['model'])
    else:
        print('Error: model not found!')

    return model

def main():

    use_gpu = True
    model_name = args.model
    model_quantized = load_model(name=model_name, quantized=False)
    print("victim model: ", args.model)

    if model_name == 'inceptionv3':
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    '''ImageNet validation set'''
    # val_dataset = torchvision.datasets.ImageFolder(root=args.val_dir, transform=preprocess)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    '''1000 clean images'''
    # print('---------------- test on clean data ----------------')
    # clean_dataset = DataSet(img_dir=args.clean_dir, transform=preprocess)
    # clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=args.batch_size,
    #                                            shuffle=False, num_workers=args.workers)
    # validate(clean_loader, model_quantized, batch_size=args.batch_size)
    # print('----------------------------------------------------')

    '''1000 adv images'''
    print('----------------- test on adv data -----------------')
    print('test data: ', args.adv_dir)
    adv_dataset = DataSet(img_dir=args.adv_dir, transform=preprocess, png=False)
    adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.workers)
    validate(adv_loader, model_quantized, batch_size=args.batch_size, use_gpu=use_gpu)
    print('----------------------------------------------------')

    '''1000 offcial adv images'''
    print('----------------- test on official adv data -----------------')
    print('test data: ', args.official_adv_dir)
    adv_dataset = DataSet(img_dir=args.official_adv_dir, transform=preprocess, png=False)
    adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers)
    validate(adv_loader, model_quantized, batch_size=args.batch_size, use_gpu=use_gpu)
    print('----------------------------------------------------')


    return



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()