import os
import argparse
import warnings
import numpy as np
from sklearn.neighbors import KernelDensity
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys

sys.path.append('/mnt/cephfs/home/zhangshuhai/DiffPure')
#
from utils import get_image_classifier,load_detection_data
#
sys.path.append('/mnt/cephfs/home/zhangshuhai/DiffPure/KD-BU/detect')
from util import (get_dataloader, get_model, set_seed,
                  get_noisy_loader, get_mc_predictions,
                  get_deep_representations, score_samples, normalize,
                  train_lr, compute_roc, get_adv_loader2, evaluate, get_subsample_loader, 
                  get_prediction)


from torch.utils.data import DataLoader, Subset, Dataset
import torch
import torchvision
import torch.nn as nn
import torchattacks
from tqdm import tqdm

# Optimal KDE bandwidths that were determined from CV tuning
BANDWIDTHS = {'mnist': 3.79, 'cifar': 0.26, 'svhn': 1.00}

def main(args):
    set_seed(0)
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    # assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw', 'all'], \
    #     "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
    #     "'jsma' or 'cw'"
    # assert os.path.isfile(f'./checkpoint/{args.dataset}.pth'), \
    #     'model file not found... must first train model using train_model.py.'
    # assert os.path.isdir(f'./adv_data/{args.dataset}'), \
        # 'adversarial sample file not found... must first craft adversarial ' \
        # 'samples using craft_adv_samples.py'
    print('Loading the data and model...')
    
    # Load the model
    criterion = nn.CrossEntropyLoss()
    model = get_image_classifier(args.classifier_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Used device : {device}')
    model.to(device)
    
    # Load the dataset
    train_loader, valid_loader, test_loader = get_dataloader(args)
    
    # Check attack type, select adversarial and noisy samples accordingly
    print('Loading noisy and adversarial samples...')
    # if args.attack == 'all':
    #     # TODO: implement 'all' option
    #     #X_test_adv = ...
    #     #X_test_noisy = ...
    #     raise NotImplementedError("'All' types detector not yet implemented.")
    # else:
    #     # Load adversarial samples
    #     test_adv_loader = get_adv_loader(args, args.dataset, args.attack)
    #     # Craft an equal number of noisy samples
    #     test_noisy_loader = get_noisy_loader(args, test_loader.dataset, test_adv_loader.dataset, args.dataset, args.attack)
    
    if args.dataset == 'cifar':
        args.num_classes = 10
        mean = torch.from_numpy(np.array([x / 255
			for x in [125.3, 123.0, 113.9]])).view(1,3,1,1).cuda().float()
        std = torch.from_numpy(np.array([x / 255
			for x in [63.0, 62.1, 66.7]])).view(1,3,1,1).cuda().float()
    
    from ensattack import ens_attack
    attack_methods = args.attack_methods
    print('reset the mean 0 and std 1')
    mean = mean - mean
    std = std / std
    
    # assert  not (args.detection_ensattack_norm_flag==True and args.single_vector_norm_flag==True)
    for attack_method in attack_methods:
        top1_counter = 0
        top5_counter = 0
        top1_clean_counter = 0
        top5_clean_counter = 0
        num_samples = 0
        X_adv = []
        Y = []
        for i, (x, y) in enumerate(test_loader):
            if i>2:
                break
            x = x.to(device)
            y = y.to(device)

            if attack_method == 'MM_Attack' or attack_method == 'AA_Attack':
                output = model(x)
                correct_batch = y.eq(output.max(dim=1)[1])
                x = x[correct_batch]
                y = y[correct_batch]
            
            x_adv, top1, top5 = ens_attack(x, y, model, mean, std, args, attack_method)
            top1_counter += top1 * x.shape[0]
            top5_counter += top5 * x.shape[0]
            top1_clean_counter += x.shape[0]
            top5_clean_counter += x.shape[0]
            num_samples += x.shape[0]          
            
            X_adv.append(x_adv)
            Y.append(y)

        # Load adversarial samples
        # X_adv = torch.from_numpy(np.asarray(X_adv))
        # Y = torch.from_numpy(np.asarray(Y))
        test_adv_loader = get_adv_loader2(args, torch.cat(X_adv,dim=0), torch.cat(Y,dim=0))
        
        print(f'attack_method: {attack_method}, robust accuracy: top1:{top1_counter/num_samples}--top5:{top5_counter/num_samples}')
        print(f'top1_clean:{top1_clean_counter/num_samples}--top5_clean:{top5_clean_counter/num_samples}')

        # Craft an equal number of noisy samples
        args.attack = attack_method
        # test_noisy_loader = get_noisy_loader(args, test_loader.dataset, test_adv_loader.dataset, args.dataset, args.attack)
        test_noisy_loader = get_noisy_loader(args, test_loader.dataset, test_adv_loader.dataset, args.dataset, attack_method)
        
        # Check model accuracies on each sample type
        for s_type, loader in zip(['normal', 'noisy', 'adversarial'],
                                [test_loader, test_noisy_loader, test_adv_loader]):
            if s_type == 'normal' :
                _, acc, pred, true = evaluate(model, loader, criterion, device, return_pred=True)
                print("Model accuracy on the %s test set: %0.2f%%" %
                (s_type, acc))
            else :
                _, acc = evaluate(model, loader, criterion, device)
                print("Model accuracy on the %s test set: %0.2f%%" %
                (s_type, acc))
                
                # Compute and display average perturbation sizes
                advs = torch.stack([x[0] for x in loader.dataset])
                l2_diff = np.linalg.norm(
                    advs.reshape((len(advs), -1)) -
                    (test_loader.dataset.data/255).reshape((len(test_loader.dataset), -1)),
                    axis=1
                ).mean()
                print("Average L-2 perturbation size of the %s test set: %0.2f" %
                    (s_type, l2_diff))
                
        # Refine the normal, noisy and adversarial sets to only include samples for
        # which the original version was correctly classified by the model
        inds_correct = torch.where((true==pred))[0]
        test_loader = get_subsample_loader(args, test_loader, inds_correct)
        test_noisy_loader = get_subsample_loader(args, test_noisy_loader, inds_correct)
        test_adv_loader = get_subsample_loader(args, test_adv_loader, inds_correct)
        
        # test_loader.dataset = Subset(test_loader.dataset, inds_correct)
        # test_noisy_loader.dataset = Subset(test_noisy_loader.dataset, inds_correct)
        # test_adv_loader.dataset = Subset(test_adv_loader.dataset, inds_correct)
        
        ## Get Bayesian uncertainty scores
        print('Getting Monte Carlo dropout variance predictions...')
        uncerts_normal = get_mc_predictions(model, test_loader, device).var(axis=0).mean(axis=1)
        uncerts_noisy = get_mc_predictions(model, test_noisy_loader, device).var(axis=0).mean(axis=1)
        uncerts_adv = get_mc_predictions(model, test_adv_loader, device).var(axis=0).mean(axis=1)

        print('Mean Uncertainty of noraml data :', uncerts_normal.mean())
        print('Mean Uncertainty of noisy data :', uncerts_noisy.mean())
        print('Mean Uncertainty of adv data :', uncerts_adv.mean())
        
        train_loader = DataLoader(dataset=train_loader.dataset, 
                            batch_size=args.batch_size,
                            shuffle=False, drop_last=False)

        ## Get KDE scores
        # Get deep feature representations
        model.eval()
        print('Getting deep feature representations...')
        X_train_features = get_deep_representations(model, train_loader, device)
        X_test_normal_features = get_deep_representations(model, test_loader, device)
        X_test_noisy_features  = get_deep_representations(model, test_noisy_loader, device)
        X_test_adv_features  = get_deep_representations(model, test_adv_loader, device)

        print('Mean KDE of train data :', X_train_features.mean())
        print('Mean KDE of test normal data :', X_test_normal_features.mean())
        print('Mean KDE of noisy data :', X_test_noisy_features.mean())
        print('Mean KDE of adv test data :', X_test_adv_features.mean())
        
        # Train one KDE per class
        print('Training KDEs...')
        # num_of_class = len(set(test_loader.dataset.dataset.targets.numpy()))
        num_of_class = 10
        class_inds = {}
        for i in range(num_of_class):
            if args.dataset != 'svhn' :
                class_inds[i] = np.where(np.array(train_loader.dataset.targets) == i)[0]
            else :
                class_inds[i] = np.where(np.array(train_loader.dataset.labels) == i)[0]
        kdes = {}
        warnings.warn("Using pre-set kernel bandwidths that were determined "
                    "optimal for the specific CNN models of the paper. If you've "
                    "changed your model, you'll need to re-optimize the "
                    "bandwidth.")
        for i in range(num_of_class):
            
            kdes[i] = KernelDensity(kernel='gaussian', bandwidth=BANDWIDTHS[args.dataset]).fit(X_train_features[class_inds[i]])

        # Get model predictions
        print('Computing model predictions...')
        model.eval()
        preds_test_normal = get_prediction(model, test_loader, device).argmax(1)
        preds_test_noisy = get_prediction(model, test_noisy_loader, device).argmax(1)
        preds_test_adv = get_prediction(model, test_adv_loader, device).argmax(1)
        
        # Get density estimates
        print('computing densities...')
        densities_normal = score_samples(
            kdes,
            X_test_normal_features,
            preds_test_normal,
            n_jobs=8
        )
        densities_noisy = score_samples(
            kdes,
            X_test_noisy_features,
            preds_test_noisy,
            n_jobs=8
        )
        densities_adv = score_samples(
            kdes,
            X_test_adv_features,
            preds_test_adv,
            n_jobs=8
        )

        ## Z-score the uncertainty and density values
        uncerts_test_z, uncerts_noisy_z, uncerts_adv_z = normalize(
            uncerts_normal,
            uncerts_noisy,
            uncerts_adv
        )
        densities_test_z, densities_noisy_z, densities_adv_z = normalize(
            densities_normal,
            densities_noisy,
            densities_adv
        )

        ## Build detector
        values, labels, lr = train_lr(
            densities_pos=densities_adv_z,
            densities_neg=np.concatenate((densities_test_z, densities_noisy_z)),
            uncerts_pos=uncerts_adv_z,
            uncerts_neg=np.concatenate((uncerts_test_z, uncerts_noisy_z))
        )

        ## Evaluate detector
        # Compute logistic regression model predictions
        probs = lr.predict_proba(values)[:, 1]
        # Compute AUC
        n_samples = len(densities_adv)
        # The first 2/3 of 'probs' is the negative class (normal and noisy samples),
        # and the last 1/3 is the positive class (adversarial samples).
        fpr, tpr, auc_score = compute_roc(
            probs_neg=probs[:2*n_samples],
            probs_pos=probs[2*n_samples:], plot=True
        )
        print('Detector ROC-AUC score: %0.4f' % auc_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        default='cifar',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
       type=str
    )
    # parser.add_argument(
    #     '-a', '--attack',
    #     help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma' 'cw' "
    #          "or 'all'",
    #     required=True, type=str
    # )
    parser.add_argument(
        '-b', '--batch_size',
        default=256,
        help="The batch size to use for training.",
        required=False, type=int
    )

    ##
    parser.add_argument('--classifier_name', help="which classifier to use", default='cifar10-wideresnet-28-10', type=str)
    parser.add_argument('--attack_methods', type=str, nargs='+', default=['PGD','BIM_L2','FGSM', 'CW', 'BIM',  'MIM', 'TIM', 'DI_MIM','FGSM_L2',  'PGD_L2', 'MM_Attack', 'AA_Attack'])
    parser.add_argument('--domain', type=str, default='cifar10', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--datapath', type=str,  default='/mnt/cephfs/home/zhangshuhai/DiffPure/dataset')
    parser.add_argument( '--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument( '--num_sub', type=int, default=256, help='imagenet subset')
    parser.add_argument( '--random', default=True, help='random initialization for PGD')
    parser.add_argument('--epsilon', default=0.01568, type=float, help='perturbation')
    parser.add_argument('--num-steps', default=5, type=int,help='perturb number of steps')

    # parser.set_defaults(batch_size=256)
    args = parser.parse_args()

    args.step_size_adv = args.epsilon / args.num_steps

    main(args)