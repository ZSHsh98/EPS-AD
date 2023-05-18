import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
import datetime
from models import *
import numpy as np
import attack_generator as attack
from utils import Logger
import os

parser = argparse.ArgumentParser(description='PyTorch MM Attack')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="resnet18",help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--dataset', type=str, default='cifar10', help="choose from cifar10,svhn,cifar100")
parser.add_argument('--num_classes', type=int, default=10, help='num classes')
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")
parser.add_argument('--omega', type=float, default=0.0, help="random sample parameter")
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--out_dir',type=str,default='./result' ,help='dir of output')
parser.add_argument('--resume', type=str, default='./CE_resnet18/61epochpoint.pth.tar', help='whether to resume training')
parser.add_argument('--model_resume',type=str,default='CE' ,help='type of the model')

# args for mm attack
parser.add_argument('--mode',type=str,default='attack' ,help='decide to compare or just use mm attack, choose from compare,attack')
parser.add_argument('--perturb_steps', type=int, default=20, help='perturb steps for attack')
parser.add_argument('--eps', type=float, default=0.031, help='eps bound for attack')
parser.add_argument('--k', type=int, default=3, help='top k classes choose for mm attack')


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
    model = Wide_ResNet(depth=args.depth, num_classes=10, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(args.depth, args.width_factor, args.drop_rate)
if args.model_resume == 'CE':
    model = torch.nn.DataParallel(model)
print(net)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

start_epoch = 0
# Resume
title = 'MM Attack'
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

test_loader_all = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=2)

# Evalutions
if args.mode=='compare':
    loss, test_nat_acc = attack.eval_clean(model, test_loader)
    print('test nat acc:')
    print(test_nat_acc)

    loss, pgd20_acc, pgd20_time = attack.eval_robust(model,test_loader, perturb_steps=20, epsilon=0.031, step_size=0.003,loss_fn="cent",category="Madry",rand_init=True,num_classes=args.num_classes)
    print('pgd20 acc:')
    print(pgd20_acc)
    print('pgd20 time:')
    print(pgd20_time)

    loss, cw_acc, cw_time = attack.eval_robust(model,test_loader, perturb_steps=20, epsilon=0.031, step_size=0.003,loss_fn="cw",category="Madry",rand_init=True,num_classes=args.num_classes)
    print('cw acc:')
    print(cw_acc)
    print('cw time:')
    print(cw_time)

    loss, autopgd_ce_acc, autopgd_ce_time  = attack.eval_robust_auto(model, test_loader, perturb_steps=100, epsilon=0.031, loss_fn="ce")
    print('apgd ce acc:')
    print(autopgd_ce_acc)
    print('apgd ce time:')
    print(autopgd_ce_time)

    loss, autopgd_dlr_acc, autopgd_dlr_time  = attack.eval_robust_auto(model, test_loader, perturb_steps=100, epsilon=0.031, loss_fn="dlr")
    print('apgd dlr acc:')
    print(autopgd_dlr_acc)
    print('apgd dlr time:')
    print(autopgd_dlr_time)

    loss, autopgd_fab_acc, autopgd_fab_time  = attack.eval_robust_auto_fab(model, test_loader, perturb_steps=100, epsilon=0.031)
    print('apgd fab acc:')
    print(autopgd_fab_acc)
    print('apgd fab time:')
    print(autopgd_fab_time)

    loss, autopgd_square_acc, autopgd_square_time  = attack.eval_robust_auto_square(model, test_loader, epsilon=0.031)
    print('apgd square acc:')
    print(autopgd_square_acc)
    print('apgd square time:')
    print(autopgd_square_time)

    loss, test_mm3_acc, mm3_time = attack.eval_robust_mm_sequential(model, test_loader_all, perturb_steps=20, epsilon=0.031, loss_fn='mm', k=3)
    print('MM3 acc:')
    print(test_mm3_acc)
    print('MM3 time:')
    print(mm3_time)

    loss, test_mm9_acc, mm9_time = attack.eval_robust_mm_sequential(model, test_loader_all, perturb_steps=20, epsilon=0.031, loss_fn='mm', k=9)
    print('MM9 acc:')
    print(test_mm9_acc)
    print('MM9 time:')
    print(mm9_time)

    loss, test_mm_plus_acc, mm_plus_time = attack.eval_robust_mm_sequential(model, test_loader_all, perturb_steps=100, epsilon=0.031, loss_fn='mm', k=9)
    print('MM_plus acc:')
    print(test_mm_plus_acc)
    print('MM_plus time:')
    print(mm_plus_time)

elif args.mode=='attack':
    loss, test_mm_acc, mm_time = attack.eval_robust_mm_sequential(model, test_loader_all, perturb_steps=args.perturb_steps, epsilon=args.eps, loss_fn='mm', k=args.k)
    print('MM attack acc:')
    print(test_mm_acc)
    print('MM attack time:')
    print(mm_time)