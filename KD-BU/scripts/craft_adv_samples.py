import os
import argparse
import numpy as np

import sys
sys.path.append('./detect')

from util import get_dataloader, get_model, set_seed, evaluate, get_adv_loader, get_noisy_loader
import torch
import torchvision
import torch.nn as nn
import torchattacks
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from tqdm import tqdm

ATTACK_PARAMS = {
    'mnist': {'eps': 0.300, 'eps_iter': 0.010},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010}
}

def craft_one_type(model, loader, dataset, attack, batch_size, criterion, device):
    if attack == 'fgsm':
        # FGSM attack
        print('Crafting fgsm adversarial samples...')
        atk = torchattacks.FGSM(model, eps=ATTACK_PARAMS[dataset]['eps']*1.3)
    elif attack in ['bim-a', 'bim-b']:
        # BIM attack
        print('Crafting %s adversarial samples...' % attack)
        atk = torchattacks.BIM(model, eps=ATTACK_PARAMS[dataset]['eps'], alpha=ATTACK_PARAMS[dataset]['eps_iter'], steps=50)
        if attack == 'bim-a':
            atk.set_mode_targeted_by_function(lambda images, labels:(labels+1)%10)
            # atk.set_mode_targeted_random()
            # atk.set_mode_targeted_least_likely(kth_min=1)
            
    elif attack == 'jsma':
        pass
        # JSMA attack
        # print('Crafting jsma adversarial samples. This may take a while...')
        # X_adv = saliency_map_method(
        #     sess, model, X, Y, theta=1, gamma=0.1, clip_min=0., clip_max=1.
        # )
    else:
        # TODO: CW attack
        atk = torchattacks.CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
    
    i=0
    model.eval()
    true = []
    pred = []
    # if dataset == 'svhn' :
    #     classes = list(set(loader.dataset.labels))
    # else :
    #     classes = list(loader.dataset.class_to_idx.values())
    # for c in classes :
        # if not os.path.isdir(f'adv_data/{dataset}/{attack}/{c}') :
            # os.makedirs(f'adv_data/{dataset}/{attack}/{c}')
    if not os.path.isdir(f'adv_data/{dataset}') :
        os.makedirs(f'adv_data/{dataset}')
    adv_x, adv_y = [], []
    for (img, label) in tqdm(loader) :
        img = img.to(device)
        label = label.to(device)
        X_adv = atk(img, label)

        outputs = model(X_adv)
        pred.append(outputs.argmax(dim=1))
        true.append(label)
        
        adv_x.append(X_adv)
        adv_y.append(label)
        # for x_adv, y_adv in zip(X_adv, label) :
        #     # pil_img = transforms.ToPILImage()(x_adv)
        #     # pil_img.save(f'adv_data/{dataset}/{attack}/{y_adv.cpu().numpy()}/{str(i)}.jpg', quality=95)
        #     save_image(x_adv, f'adv_data/{dataset}/{attack}/{y_adv.cpu().numpy()}/{str(i)}.jpg')
        #     i+=1
    adv_x = torch.cat(adv_x, 0)
    adv_y = torch.cat(adv_y, 0)
    adv_imgs = {'X':adv_x.cpu(), 'Y':adv_y.cpu()}
    torch.save(adv_imgs, f'adv_data/{dataset}/{attack}.pkl')
    
    true = torch.cat(true, dim=0)
    pred = torch.cat(pred, dim=0)
    correct_predictions = pred.eq(true).sum()
    accuracy = correct_predictions / len(loader.dataset) * 100
        
    print("Model accuracy on the adversarial test set: %0.2f%%" % (accuracy.cpu().numpy()))
    
    print('Loading noisy and adversarial samples...')

    # Load adversarial samples
    test_adv_loader = get_adv_loader(args, args.dataset, args.attack)
    # Craft an equal number of noisy samples
    test_noisy_loader = get_noisy_loader(args, loader.dataset, test_adv_loader.dataset, args.dataset, args.attack)

    # Check model accuracies on each sample type
    for s_type, ld in zip(['normal', 'noisy', 'adversarial'],
                               [loader, test_noisy_loader, test_adv_loader]):
        if s_type == 'normal' :
            _, acc, pred, true = evaluate(model, ld, criterion, device, return_pred=True)
            print("Model accuracy on the %s test set: %0.2f%%" %
              (s_type, acc))
        else :
            _, acc = evaluate(model, ld, criterion, device)
            print("Model accuracy on the %s test set: %0.2f%%" %
              (s_type, acc))
            
            # Compute and display average perturbation sizes
            advs = torch.stack([x[0] for x in ld.dataset])
            l2_diff = np.linalg.norm(
                advs.reshape((len(advs), -1)) -
                (loader.dataset.data/255).reshape((len(loader.dataset), -1)),
                axis=1
            ).mean()
            print("Average L-2 perturbation size of the %s test set: %0.2f" %
                  (s_type, l2_diff))
    
def main(args):
    set_seed(0)
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma' or 'cw'"
    assert os.path.isfile(f'./checkpoint/{args.dataset}.pth'), \
        'model file not found... must first train model using train_model.py.'
    print('Dataset: %s. Attack: %s' % (args.dataset, args.attack))
    
    _, _, test_loader = get_dataloader(args)
    criterion = nn.CrossEntropyLoss()
    model = get_model(args.dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Used device : {device}')
    model.to(device)
    
    weight = torch.load(f'./checkpoint/{args.dataset}.pth')
    model.load_state_dict(weight, strict=False)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print('test acc : {:.2f}% \t test loss : {:.4f}'.format(test_acc, test_loss))
      
    if args.attack == 'all':
        # Cycle through all attacks
        for attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw']:
            craft_one_type(model, test_loader, args.dataset, args.attack,
                           args.batch_size, criterion, device)
    else:
        # Craft one specific attack type
        craft_one_type(model, test_loader, args.dataset, args.attack,
                       args.batch_size, criterion, device)
        
    print('Adversarial samples crafted and saved to data/ subfolder.')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma', 'cw' "
             "or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=256)
    args = parser.parse_args()
    main(args)