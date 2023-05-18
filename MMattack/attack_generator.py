import numpy as np
from models import *
import PIL
from PIL import Image
from attack_apgd import APGDAttack, APGDAttack_targeted
import datetime
from fab_pt import FABAttack_PT
from square import SquareAttack
from utils import Logger
import os

# Nomal CW loss
def  cwloss(output, target,confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

# loss for MM Attack
def mm_loss(output, target, target_choose, confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)

    target_onehot = torch.zeros(target_choose.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target_choose.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)

    other = (target_var * output).sum(1)
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

# loss for MM AT
def mm_loss_train(output, target, target_choose, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)

    target_onehot = torch.zeros(target_choose.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target_choose.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)

    other = (target_var * output).sum(1)
    return other-real


# Nomal PGD
def pgd(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init,num_classes):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target,num_classes=num_classes)
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

# tpgd
def tpgd(model, data, target, epsilon, step_size, step1_size, num_steps,loss_fn,category,rand_init,k,num_classes):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    logits = model(data)
    target_onehot = torch.zeros(target.size() + (len(logits[0]),))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    index = torch.argsort(logits-10000*target_var)[:,num_classes-k:]

    x_adv_set = []
    for i in range(k):
        x_adv_0 = x_adv.clone().detach()
        for j in range(num_steps):
            x_adv_0.requires_grad_()
            output1 = model(x_adv_0)
            model.zero_grad()
            with torch.enable_grad():
                    loss_adv0 = -1. * nn.CrossEntropyLoss(reduction="sum")(output1, index[:,i])
            loss_adv0.backward()
            eta = step_size * x_adv_0.grad.sign()
            x_adv_0 = x_adv_0.detach() + eta
            x_adv_0 = torch.min(torch.max(x_adv_0, data - epsilon), data + epsilon)
            x_adv_0 = torch.clamp(x_adv_0, 0.0, 1.0)
        x_adv_set.append(x_adv_0)
    return x_adv_set


# evaluation for clean examples
def eval_clean(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    log = 'Natrual Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

# evaluation for normal PGD and CW attack
def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, rand_init, num_classes):
    starttime = datetime.datetime.now()
    model.eval()
    test_loss = 0
    correct = 0

    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            x_adv = pgd(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=rand_init, num_classes=num_classes)
            output = model(x_adv)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    log = 'Attack Setting ==> Loss_fn:{}, Perturb steps:{}, Epsilon:{}, Step dize:{} \n Test Result ==> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(loss_fn,perturb_steps,epsilon,step_size,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    # print(log)
    test_accuracy = correct / len(test_loader.dataset)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    return test_loss, test_accuracy, time

# evaluation for mmu
def eval_robust_mmu_sequential(model, test_loader, perturb_steps, epsilon, step_size, step1_size, loss_fn, category, rand_init,k,num_classes):
    starttime = datetime.datetime.now()
    model.eval()
    test_loss = 0
    bs = 128

    with torch.no_grad():

        for data, target in test_loader:
            x_orig, y_orig = data, target

        n_batches = int(np.ceil(len(test_loader.dataset)/bs))
        robust_flags = torch.zeros(len(test_loader.dataset),dtype=torch.bool)
        for batch_idx in range(n_batches):
            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, len(test_loader.dataset))
            x = x_orig[start_idx:end_idx, :].clone().cuda()
            y = y_orig[start_idx:end_idx].clone().cuda()
            output = model(x)
            correct_batch = y.eq(output.max(dim=1)[1])
            robust_flags[start_idx:end_idx] = correct_batch.detach()
        
        robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]

        x_adv = x_orig.clone().detach()


    for i in range(k):
        num_robust = torch.sum(robust_flags).item()

        n_batches = int(np.ceil(num_robust / bs))

        robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
        if num_robust > 1:
            robust_lin_idcs.squeeze_()
                
        for batch_idx in range(n_batches):


            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, num_robust)

            batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
            if len(batch_datapoint_idcs.shape) > 1:
                batch_datapoint_idcs.squeeze_(-1)
            x = x_orig[batch_datapoint_idcs, :].clone().cuda()
            y = y_orig[batch_datapoint_idcs].clone().cuda()

            logits = model(x)
            target_onehot = torch.zeros(y.size() + (len(logits[0]),))
            target_onehot = target_onehot.cuda()
            target_onehot.scatter_(1, y.unsqueeze(1), 1.)
            target_var = Variable(target_onehot, requires_grad=False)
            index_i = torch.argsort(logits-10000*target_var)[:,-(i+1)]

            # make sure that x is a 4d tensor even if there is only a single datapoint left
            if len(x.shape) == 3:
                x.unsqueeze_(dim=0)
            
            x_adv0 = x.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, x.shape)).float().cuda() if rand_init else x.detach()
            x_adv0 = torch.clamp(x_adv0, 0.0, 1.0)
            for j in range(perturb_steps):
                x_adv0.requires_grad_()
                output = model(x_adv0)
                model.zero_grad()
                with torch.enable_grad():
                    loss_adv = mm_loss(output,y,index_i,num_classes=num_classes)
                    loss_adv.backward()
                    eta = step_size * x_adv0.grad.sign()
                    x_adv0 = x_adv0.detach() + eta
                    x_adv0 = torch.min(torch.max(x_adv0, x - epsilon), x + epsilon)
                    x_adv0 = torch.clamp(x_adv0, 0.0, 1.0)


            output = model(x_adv0)
            false_batch = ~y.eq(output.max(dim=1)[1]).cuda()
            non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
            robust_flags[non_robust_lin_idcs] = False
            # x_adv[non_robust_lin_idcs] = x_adv[false_batch].detach().cuda()

        robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    return test_loss, robust_accuracy, time

# evaluation for autoattack
def eval_robust_auto(model, test_loader, perturb_steps, epsilon, loss_fn):
    starttime = datetime.datetime.now()
    apgd = APGDAttack(model, n_restarts=1, n_iter=perturb_steps, verbose=False,
                eps=epsilon, norm='Linf', eot_iter=1, rho=.75, seed=1, device='cuda')
    
    # apgd = APGDAttack_targeted(model, n_restarts=1, n_iter=100, verbose=False,
    #             eps=epsilon, norm='Linf', eot_iter=1, rho=.75, seed=0, device='cuda')
    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            apgd.loss = loss_fn
            x_adv = apgd.perturb(data, target)
            output = model(x_adv)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    return test_loss, test_accuracy, time

# evaluation for fab
def eval_robust_auto_fab(model, test_loader, perturb_steps, epsilon):
    starttime = datetime.datetime.now()
    fab = FABAttack_PT(model, n_restarts=1, n_iter=perturb_steps, eps=epsilon, seed=1,
                norm='Linf', verbose=False, device='cuda')
    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            fab.targeted = False
            x_adv = fab.perturb(data, target)
            output = model(x_adv)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    return test_loss, test_accuracy, time

# evaluation for square
def eval_robust_auto_square(model, test_loader, epsilon):
    starttime = datetime.datetime.now()
    square = SquareAttack(model, p_init=.8, n_queries=5000, eps=epsilon, norm='Linf',
            n_restarts=1, seed=1, verbose=False, device='cuda', resc_schedule=False)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            x_adv = square.perturb(data, target)
            output = model(x_adv)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    return test_loss, test_accuracy, time

def eval_robust_auto_topk_target(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, rand_init, k, num_classes):
    starttime = datetime.datetime.now()
    
    apgd = APGDAttack_targeted(model, n_restarts=1, n_iter=perturb_steps, verbose=False,
                eps=epsilon, norm='Linf', eot_iter=1, rho=.75, seed=1, device='cuda')

    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()


            apgd.loss = loss_fn
            apgd.n_target_classes = k
            x_adv = apgd.perturb(data, target)

            output = model(x_adv)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    return test_loss, test_accuracy, time

def eval_robust_mm_sequential(model, test_loader, perturb_steps, epsilon, loss_fn, k):
    starttime = datetime.datetime.now()
    model.eval()
    test_loss = 0
    bs = 128

    apgd = APGDAttack_targeted(model, n_restarts=1, n_iter=perturb_steps, verbose=False,
                eps=epsilon, norm='Linf', eot_iter=1, rho=.75, seed=1, device='cuda')

    with torch.no_grad():

        for data, target in test_loader:
            x_orig, y_orig = data, target

        n_batches = int(np.ceil(len(test_loader.dataset)/bs))
        robust_flags = torch.zeros(len(test_loader.dataset),dtype=torch.bool)
        for batch_idx in range(n_batches):
            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, len(test_loader.dataset))
            x = x_orig[start_idx:end_idx, :].clone().cuda()
            y = y_orig[start_idx:end_idx].clone().cuda()
            output = model(x)
            correct_batch = y.eq(output.max(dim=1)[1])
            robust_flags[start_idx:end_idx] = correct_batch.detach()
        
        robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]

        x_adv = x_orig.clone().detach()


    for i in range(k):
        num_robust = torch.sum(robust_flags).item()

        n_batches = int(np.ceil(num_robust / bs))

        robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
        if num_robust > 1:
            robust_lin_idcs.squeeze_()
                
        for batch_idx in range(n_batches):


            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, num_robust)

            batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
            if len(batch_datapoint_idcs.shape) > 1:
                batch_datapoint_idcs.squeeze_(-1)
            x = x_orig[batch_datapoint_idcs, :].clone().cuda()
            y = y_orig[batch_datapoint_idcs].clone().cuda()

            # make sure that x is a 4d tensor even if there is only a single datapoint left
            if len(x.shape) == 3:
                x.unsqueeze_(dim=0)
            
            apgd.n_target_classes = 1
            apgd.loss = loss_fn
            x_adv0 = apgd.perturb(x, y, i=i)

            output = model(x_adv0)
            false_batch = ~y.eq(output.max(dim=1)[1]).cuda()
            non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
            robust_flags[non_robust_lin_idcs] = False

            # x_adv[non_robust_lin_idcs] = x_adv[false_batch].detach().cuda()
                
                
        robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    return test_loss, robust_accuracy, time

# evaluation for tpgd
def eval_robust_tpgd(model, test_loader, perturb_steps, epsilon, step_size, step1_size, loss_fn, category, rand_init,k,num_classes):
    starttime = datetime.datetime.now()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            x_adv_set = tpgd(model,data,target,epsilon,step_size,step1_size,perturb_steps,loss_fn,category,rand_init=rand_init,k=k,num_classes=num_classes)
            d = torch.tensor([[True] for x in range(len(target))]).cuda()
            for i in range(k):
                output = model(x_adv_set[i])
                pred = output.max(1, keepdim=True)[1]
                a = pred.eq(target.view_as(pred))
                d = d & a
            correct += d.sum()
        correct = correct.cpu().numpy()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    return test_loss, test_accuracy, time

def mmu_at_train(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init,k,num_classes):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    logits = model(data)
    target_onehot = torch.zeros(target.size() + (len(logits[0]),))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    index = torch.argsort(logits-10000*target_var)[:,num_classes-k:]

    x_adv_set = []
    loss_set = []
    for i in range(k):
        x_adv_0 = x_adv.clone().detach()
        for j in range(num_steps):
            x_adv_0.requires_grad_()
            output1 = model(x_adv_0)
            model.zero_grad()
            with torch.enable_grad():
                    loss_adv0 = mm_loss(output1,target,index[:,i],num_classes=num_classes)
            loss_adv0.backward()
            eta = step_size * x_adv_0.grad.sign()
            x_adv_0 = x_adv_0.detach() + eta
            x_adv_0 = torch.min(torch.max(x_adv_0, data - epsilon), data + epsilon)
            x_adv_0 = torch.clamp(x_adv_0, 0.0, 1.0)


        pipy = mm_loss_train(model(x_adv_0),target,index[:,i],num_classes=num_classes)
        loss_set.append(pipy.view(len(pipy),-1))
        x_adv_set.append(x_adv_0)

    loss_pipy = loss_set[0]     
    for i in range(k-1):
        loss_pipy = torch.cat((loss_pipy, loss_set[i+1]),1)
    
    index_choose = torch.argsort(loss_pipy)[:,-1]

    adv_final = torch.zeros(x_adv.size()).cuda()
    for i in range(len(index_choose)):
        adv_final[i,:,:,:] = x_adv_set[index_choose[i]][i]

    return adv_final
