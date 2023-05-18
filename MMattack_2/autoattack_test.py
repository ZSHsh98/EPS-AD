from autoattack import AutoAttack
import os
import argparse
from numpy.core.numeric import True_
import torchvision
import torch.optim as optim
from torchvision import transforms
import datetime
from models import *
import numpy as np
import attack_generator as attack
from utils import Logger

parser = argparse.ArgumentParser(description='PyTorch AutoAttack')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="resnet18",help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--dataset', type=str, default='cifar10', help="choose from cifar10,svhn")
parser.add_argument('--num_classes', type=int, default=10, help='num classes')
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")
parser.add_argument('--omega', type=float, default=0.0, help="random sample parameter")
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--out_dir',type=str,default='./result' ,help='dir of output')
parser.add_argument('--resume', type=str, default='./CE_resnet18/61epochpoint.pth.tar', help='whether to resume training, default: None')
parser.add_argument('--model_resume',type=str,default='CE' ,help='type of the model')

args = parser.parse_args()
print(args)

# settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 60:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

print('==> Load Test Data')
if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "svhn":
    trainset = torchvision.datasets.SVHN(root='../data', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.SVHN(root='../data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

print('==> Load Model')
if args.net == "smallcnn":
    model = SmallCNN().cuda()
    net = "smallcnn"
if args.net == "resnet18":
    model = ResNet18(args.num_classes).cuda()
    net = "resnet18"
if args.net == "WRN":
    model = Wide_ResNet(depth=args.depth, num_classes=args.num_classes, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(args.depth, args.width_factor, args.drop_rate)
if args.model_resume == 'CE':
    model = torch.nn.DataParallel(model)
print(net)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

start_epoch = 0
# Resume
title = 'AutoAttack'
if args.resume:
    if args.model_resume == 'CE':
        # resume directly point to checkpoint.pth.tar e.g., --resume='./out-dir/checkpoint.pth.tar'
        print ('==> Adversarial Training Resuming from checkpoint ..')
        print(args.resume)
        assert os.path.isfile(args.resume)
        out_dir = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title, resume=True)
    elif args.model_resume == 'TRADES':
        checkpoint = torch.load(args.resume)
        start_epoch = 76
        model.load_state_dict(checkpoint)
        
else:
    print('==> Adversarial Training')
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title)
    logger_test.set_names(['Epoch', 'Natural Test Acc', 'train_loss', 'PGD20 Acc', 'cw_acc', 'cw_alter_acc'])


loss, test_nat_acc = attack.eval_clean(model, test_loader, args.num_classes)
print('nat acc:')
print(test_nat_acc)

starttime = datetime.datetime.now()

adversary = AutoAttack(model, norm='Linf', eps=0.031, seed=1, version='standard')

# test_loader = torch.utils.data.DataLoader(testset, batch_size=26032, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=2)

for data, target in test_loader:
    data, target = data.cuda(), target.cuda()
    x_adv = adversary.run_standard_evaluation(data, target, bs=128)

endtime = datetime.datetime.now()
total_time = (endtime - starttime).seconds

print('total time:')
print(total_time)
