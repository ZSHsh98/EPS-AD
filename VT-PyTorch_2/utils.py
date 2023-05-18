import os.path
import pandas as pd

import torch
import torch.nn as nn
import torchvision
import torchvision.models.quantization as q_models


from PIL import Image

from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(self, img_dir, label_dir='./data/val_rs.csv', transform=None, png=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.labels = pd.read_csv(self.label_dir).to_numpy()
        self.png = png



    def __getitem__(self, index):
        file_name, label = self.labels[index]
        label = torch.tensor(label) - 1
        file_dir = os.path.join(self.img_dir, file_name)
        if self.png:
            file_dir = jpeg2png(file_dir)
        img = Image.open(file_dir).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label


    def __len__(self):
        return len(self.labels)


def validate(val_loader, model, batch_size, use_gpu=True):

    # switch to evaluate mode
    if use_gpu:
        model = model.cuda()
    model.eval()
    acc_top1 = []
    acc_top5 = []
    for i, (input, target) in enumerate(val_loader):
        if use_gpu:
            input = input.cuda()
            target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            # measure accuracy
            pred_top1 = output.topk(k=1, largest=True).indices
            pred_top5 = output.topk(k=5, largest=True).indices

            if pred_top1.dim() >= 2:
                pred_top1 = pred_top1.squeeze()
            acc_top1.append((pred_top1 == target).sum().item())
            acc_top5.append((pred_top5 == target.unsqueeze(dim=1).expand(-1, 5)).sum().item())
            # print('test phase, break now...')
            # break
            if i % 100 == 99:
                print('i = %d' % (i + 1))

    print('top1 accuracy in overall: ', sum(acc_top1) / (len(val_loader) * batch_size))
    print('top5 accuracy in overall: ', sum(acc_top5) / (len(val_loader) * batch_size))

    return


def save_images(adv, i, batch_size, output_dir, png=True):
    '''
    save the adversarial images
    :param adv: adversarial images in [0, 1]
    :param i: batch index of images
    :return:
    '''
    dest_dir = output_dir
    labels = pd.read_csv('./data/val_rs.csv').to_numpy()
    base_idx = i * batch_size
    for idx, img in enumerate(adv):
        fname = labels[idx + base_idx][0]
        dest_name = os.path.join(dest_dir, fname)
        if png:
            dest_name = jpeg2png(dest_name)
        torchvision.utils.save_image(img, dest_name)
    return


def jpeg2png(name):
    name_list = list(name)
    name_list[-4:-1] = 'png'
    name_list.pop(-1)
    return ''.join(name_list)

# copy from advertorch
class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

def load_model(name):

    model_dir_fp = './checkpoints/fp_models'
    model_map = model_map_fp
    model_dir = os.path.join(model_dir_fp, model_map[name])

    if name == 'resnet18':
        model = q_models.resnet18(pretrained=False, quantize=False)
        model.load_state_dict(torch.load(model_dir))
    elif name == 'resnet50':
        model = q_models.resnet50(pretrained=False, quantize=False)
        model.load_state_dict(torch.load(model_dir))
    elif name == 'googlenet':
        model = q_models.googlenet(pretrained=False, quantize=False)
        model.load_state_dict(torch.load(model_dir))
    elif name == 'inceptionv3':
        model = q_models.inception_v3(pretrained=False, quantize=False)
        model.load_state_dict(torch.load(model_dir))

    elif name == 'resnext101':
        model = q_models.resnext101_32x8d(pretrained=False, quantize=False)
        model.load_state_dict(torch.load(model_dir))
    elif name == 'shufflenetv2':
        model = q_models.shufflenet_v2_x1_0(pretrained=False, quantize=False)
        model.load_state_dict(torch.load(model_dir))
    elif name == 'mobilenetv2':
        model = q_models.mobilenet_v2(pretrained=False, quantize=False)
        model.load_state_dict(torch.load(model_dir))

    return model


model_map_fp = {
    'googlenet': 'googlenet-1378be20.pth',
    'inceptionv3': 'inception_v3_google-1a9a5a14.pth',
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnext101': 'resnext101_32x8d-8ba56ff5.pth',
    'shufflenetv2': 'shufflenetv2_x1-5666bf0f80.pth',
    'mobilenetv2': 'mobilenet_v2-b0353104.pth',
}