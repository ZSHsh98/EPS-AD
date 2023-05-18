import os
import argparse
from numpy.core import numeric
import torchvision
import torch.optim as optim
from torchvision import transforms
import datetime
from models import *
import numpy as np
from utils import Logger
import attack_generator as attack

parser = argparse.ArgumentParser(description='PyTorch MM_U AT')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=0.007, help='step size')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="resnet18",
                    help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--dataset', type=str, default='cifar10', help="choose from cifar10,svhn")
parser.add_argument('--num_classes', type=int, default=10, help='num classes')
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")
parser.add_argument('--omega', type=float, default=0.001, help="random sample parameter for adv data generation")
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--out_dir', type=str, default='./MMU_AT_resnet18', help='dir of output')
parser.add_argument('--resume', type=str, default='', help='whether to resume training, default: None')

args = parser.parse_args()
print(args)
# training settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def train(model, train_loader, optimizer):
    starttime = datetime.datetime.now()
    loss_sum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()


        output_adv = attack.mmu_at_train(model,data,target,epsilon=args.epsilon,step_size=args.step_size,step1_size=args.step_size,
                        num_steps=args.num_steps,loss_fn='cent',category="Madry",rand_init=args.rand_init,k=3,num_classes=10)

        output_natural = data
        output_target = target
        
        model.train()
        optimizer.zero_grad()
        output = model(output_adv)
        
        # calculate loss
        loss = nn.CrossEntropyLoss(reduction='mean')(output, output_target)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds

    return time, loss_sum


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 60:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 110:
        lr = args.lr * 0.005
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
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
if args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
if args.dataset == "svhn":
    trainset = torchvision.datasets.SVHN(root='../data', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.SVHN(root='../data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

print('==> Load Model')
if args.net == "smallcnn":
    model = SmallCNN().cuda()
    net = "smallcnn"
if args.net == "resnet18":
    model = ResNet18(num_classes=args.num_classes).cuda()
    net = "resnet18"
if args.net == "WRN":
  # e.g., WRN-34-10
    model = Wide_ResNet(depth=args.depth, num_classes=args.num_classes, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(args.depth, args.width_factor, args.drop_rate)
if args.net == 'WRN_madry':
  # e.g., WRN-32-10
    model = Wide_ResNet_Madry(depth=args.depth, num_classes=args.num_classes, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    net = "WRN_madry{}-{}-dropout{}".format(args.depth, args.width_factor, args.drop_rate)
print(net)

model = torch.nn.DataParallel(model)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

start_epoch = 0
# Resume
title = 'mmu at'
if args.resume:
    # resume directly point to checkpoint.pth.tar e.g., --resume='./out-dir/checkpoint.pth.tar'
    print('==> Training Resuming from checkpoint ..')
    print(args.resume)
    assert os.path.isfile(args.resume)
    out_dir = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title, resume=True)
else:
    print('==> Adversarial Training')
    logger_test = Logger(os.path.join(args.out_dir, 'log_results.txt'), title=title)
    logger_test.set_names(['Epoch', 'Natural Test Acc', 'train_loss', 'PGD20 Acc', 'CW Acc'])


for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch + 1)
    train_time, train_loss = train(model, train_loader, optimizer)

    ## Evalutions
    loss, test_nat_acc = attack.eval_clean(model, test_loader, args.num_classes)

    loss, test_pgd20_acc, pgd20_time = attack.eval_robust(model,test_loader, perturb_steps=20, epsilon=0.031, step_size=0.003,loss_fn="cent",category="Madry",rand_init=True,num_classes=args.num_classes)

    loss, test_cw_acc, cw_time = attack.eval_robust(model,test_loader, perturb_steps=20, epsilon=0.031, step_size=0.003,loss_fn="cw",category="Madry",rand_init=True,num_classes=args.num_classes)

    print(
        'Epoch: [%d | %d] | Train Time: %.4f s | Natural Test Acc %.4f | test_pgd20_acc %.4f | test_cw_acc %.4f |\n' % (
            epoch + 1,
            args.epochs,
            train_time,
            test_nat_acc,
            test_pgd20_acc,
            test_cw_acc)
    )

    logger_test.append([epoch + 1, test_nat_acc, train_loss, test_pgd20_acc, test_cw_acc])

    if epoch+1 == 61:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'test_nat_acc': test_nat_acc,
            'test_pgd20_acc': test_pgd20_acc,
            'optimizer': optimizer.state_dict(),
        },filename = '61epochpoint.pth.tar')

    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'test_nat_acc': test_nat_acc,
            'test_pgd20_acc': test_pgd20_acc,
            'optimizer': optimizer.state_dict(),
        })