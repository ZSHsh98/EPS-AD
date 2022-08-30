import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale

from PIL import Image
from collections import Counter

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.utils as vutils
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchattacks
import random
import torch.backends.cudnn as cudnn

STDEVS = {
    'mnist': {'fgsm': 0.310, 'bim-a': 0.220, 'bim-b': 0.230},
    'cifar': {'fgsm': 0.050, 'bim-a': 0.009, 'bim-b': 0.039},
    'svhn': {'fgsm': 0.175, 'bim-a': 0.070, 'bim-b': 0.105}
}

# Set random seed
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataloader(args):
    transformer = __get_transformer(args)
    dataset = __get_dataset_name(args)
    trn_loader, dev_loader, tst_loader = __get_loader(args, dataset, transformer)

    return trn_loader, dev_loader, tst_loader

def get_subsample_loader(args, loader, indices):
    subsample_loader = torch.utils.data.DataLoader(
        Subset(loader.dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )
    return subsample_loader

def __get_loader(args, data_name, transformer):
    root = os.path.join('./data',data_name)
    data_path = os.path.join(root, args.dataset.lower())
    dataset = getattr(torchvision.datasets, data_name)

    # set transforms
    trn_transform = transformer
    tst_transform = transformer
    # call dataset
    # normal training set
    
    # if data_name == 'SVHN':
    #     trainset = dataset(
    #         root=data_path, download=True, split='train', transform=trn_transform
    #     )
    #     trainset, devset = torch.utils.data.random_split(
    #         trainset, [int(len(trainset) * 0.7)+1, int(len(trainset) * 0.3)]
    #     )
    #     tstset = dataset(
    #         root=data_path, download=True, split='test', transform=tst_transform
    #     )
    # else:
    #     trainset = dataset(
    #         root=data_path, download=True, train=True, transform=trn_transform
    #     )
    #     trainset, devset = torch.utils.data.random_split(
    #         trainset, [int(len(trainset) * 0.7), int(len(trainset) * 0.3)]
    #     )
    #     # validtaion, testing set
    #     tstset = dataset(
    #         root=data_path, download=True, train=False, transform=tst_transform
    #     )
    if data_name == 'SVHN':
        trainset = dataset(
            root=data_path, download=True, split='train', transform=trn_transform
        )
        tstset = dataset(
            root=data_path, download=True, split='test', transform=tst_transform
        )
    else:
        trainset = dataset(
            root=data_path, download=True, train=True, transform=trn_transform
        )
        tstset = dataset(
            root=data_path, download=True, train=False, transform=tst_transform
        )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    devloader = torch.utils.data.DataLoader(
        tstset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )
    tstloader = torch.utils.data.DataLoader(
        tstset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )
    
    print(len(trainloader.dataset), len(devloader.dataset), len(tstloader.dataset))

    return trainloader, devloader, tstloader

def __get_transformer(args):
    if args.dataset == 'mnist' :
        return transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    else : 
        return transforms.Compose([transforms.ToTensor()])

def __get_dataset_name(args):
    if args.dataset.lower() == "mnist":
        d_name = "MNIST"
    elif args.dataset.lower() == "fmnist":
        d_name = "FashionMNIST"
    elif args.dataset.lower() == "cifar":
        d_name = "CIFAR10"
    elif args.dataset.lower() == "cifar100":
        d_name = "CIFAR100"
    elif args.dataset.lower() == "svhn":
        d_name = "SVHN"
    return d_name

def get_model(dataset='mnist'):
    assert dataset in ['mnist', 'cifar', 'svhn'], \
        "dataset parameter must be either 'mnist' 'cifar' or 'svhn'"
    if dataset == 'mnist':
        # MNIST model
        model = nn.Sequential(
            nn.Conv2d(1,64,3),
            nn.ReLU(),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
            nn.Flatten(),
            nn.Linear(9216,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,10)
        )
    elif dataset == 'cifar':
        # MNIST model
        model = nn.Sequential(
            nn.Conv2d(3,32,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32,32,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(4*4*128,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,10)
        )
    elif dataset == 'svhn':
        # MNIST model
        model = nn.Sequential(
            nn.Conv2d(3,64,3),
            nn.ReLU(),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
            nn.Flatten(),
            nn.Linear(14*14*64,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,10)
        )
        
    return model

class NosiyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X_test, X_test_adv, dataset, attack):
        
        self.dataset = dataset
        self.test = X_test
        self.adv = X_test_adv
        self.attack = attack
        # self.noise = torch.randn_like(torch.tensor(X_test.data/255))*STDEVS[dataset][attack]

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        x, y = self.test[idx]
        x_adv, _ = self.adv[idx]
        org_shape = x.shape
        if self.attack in ['jsma', 'cw'] :
            nb_diff = len(torch.where(x != x_adv)[0])
            org_shape = x.shape
            x = x.view(-1)
            candidate_inds = torch.where(x < 0.99)[0]
            assert candidate_inds.shape[0] >= nb_diff
            inds = np.random.choice(candidate_inds, nb_diff)
            x[inds] = 1.0
            return x.view(org_shape), y
        else :
            noise = torch.randn_like(x)*STDEVS[self.dataset][self.attack]
            x += noise
            return torch.clip(x, min=0.0, max=1.0), y
        
def get_noisy_loader(args, X_test, X_test_adv, dataset, attack) :
    loader = torch.utils.data.DataLoader(
        NosiyDataset(X_test, X_test_adv, dataset, attack),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
        )
    return loader

class AdvDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset, attack):
        
        self.dataset = dataset
        self.attack = attack
        self.data_dict = torch.load(f'adv_data/{dataset}/{attack}.pkl')

    def __len__(self):
        return len(self.data_dict['X'])

    def __getitem__(self, idx):
        x, y = self.data_dict['X'][idx], self.data_dict['Y'][idx]
        return x, y

def get_adv_loader(args, dataset, attack) :
    loader = torch.utils.data.DataLoader(
        AdvDataset(dataset, attack),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
        )
    return loader
    
# def get_adv_loader(args, dataset, attack) :
#     transformer = __get_transformer(args)
#     adv_dataset = torchvision.datasets.ImageFolder(f"adv_data/{dataset}/{attack}", transform=transformer)
#     loader = torch.utils.data.DataLoader(
#         adv_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         drop_last=False
#         )
#     return loader

def get_prediction(model, loader, device) :
    outputs=[]
    for i, data in enumerate(loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        output = model(inputs)
        outputs.append(output.cpu().detach())
    return torch.cat(outputs, 0).numpy()

def get_mc_predictions(model, loader, device, nb_iter=50):
    model.train()
    
    def predict():
        outputs = []
        for i, data in enumerate(loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            output = model(inputs)
            outputs.append(output)

        return torch.cat(outputs, 0).cpu().numpy()

    preds_mc = []
    with torch.no_grad() :
        for i in tqdm(range(nb_iter)):
            preds_mc.append(predict())

    return np.array(preds_mc)

def get_deep_representations(model, loader, device):
    feature_extractor = list(model.modules())[0][:-2]
    outputs = []
    for data in loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        output = feature_extractor(inputs)
        outputs.append(output.cpu().detach())

    return torch.cat(outputs, 0).numpy()

def score_point(tup):
    """
    TODO
    :param tup:
    :return:
    """
    x, kde = tup

    return kde.score_samples(np.reshape(x, (1, -1)))[0]

def score_samples(kdes, samples, preds, n_jobs=-1):
    """
    TODO
    :param kdes:
    :param samples:
    :param preds:
    :param n_jobs:
    :return:
    """
    if n_jobs is not None:
        p = mp.Pool(n_jobs)
    else:
        p = mp.Pool()
    results = np.asarray(
        p.map(
            score_point,
            [(x, kdes[i]) for x, i in zip(samples, preds)]
        )
    )
    p.close()
    p.join()

    return results

def normalize(normal, adv, noisy):
    """
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))

    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:]

def train_lr(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    """
    TODO
    :param densities_pos:
    :param densities_neg:
    :param uncerts_pos:
    :param uncerts_neg:
    :return:
    """
    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])
    values_pos = np.concatenate(
        (densities_pos.reshape((1, -1)),
         uncerts_pos.reshape((1, -1))),
        axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))

    lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)

    return values, labels, lr

def compute_roc(probs_neg, probs_pos, plot=False):
    """
    TODO
    :param probs_neg:
    :param probs_pos:
    :param plot:
    :return:
    """
    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score

def evaluate(model, loader, criterion, device, return_pred=False) :
    model.eval()
    with torch.no_grad() :
        true = []
        pred = []
        test_loss = 0
        for i, data in enumerate(loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            # outputs = nn.Softmax(dim=1)(outputs)
            loss = criterion(outputs, labels)

            pred.append(outputs.argmax(dim=1))
            true.append(labels)
            test_loss += len(inputs)*loss
        true = torch.cat(true, dim=0)
        pred = torch.cat(pred, dim=0)
        correct_predictions = pred.eq(true).sum()
        accuracy = correct_predictions / len(loader.dataset) * 100
        if return_pred :
            return test_loss.cpu().numpy()/len(loader.dataset), accuracy.cpu().numpy(), pred, true
        else :
            return test_loss.cpu().numpy()/len(loader.dataset), accuracy.cpu().numpy()