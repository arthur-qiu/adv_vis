# -*- coding: utf-8 -*-
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import json
from attack_methods import pgd
from models.wrn import WideResNet
from option import BaseOptions

class PrivateOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # WRN Architecture
        self.parser.add_argument('--layers', default=28, type=int, help='total number of layers')
        self.parser.add_argument('--widen-factor', default=10, type=int, help='widen factor')
        self.parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')

# /////////////// Training ///////////////

def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    for bx, by in train_loader:
        bx, by = bx.cuda(), by.cuda()

        adv_bx = adversary_train(net, bx, by)

        # forward
        logits = net(adv_bx)

        # backward
        # scheduler.step()
        optimizer.zero_grad()
        loss = F.cross_entropy(logits, by)
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg

# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            adv_data = adversary_test(net, data, target)

            # forward
            output = net(adv_data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)

# overall_test function
def test_in_testset():
    net.eval()
    loss_avg = 0.0
    correct = 0
    adv_loss_avg = 0.0
    adv_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            adv_data = adversary_test(net, data, target)

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

            # forward
            adv_output = net(adv_data)
            adv_loss = F.cross_entropy(adv_output, target)

            # accuracy
            adv_pred = adv_output.data.max(1)[1]
            adv_correct += adv_pred.eq(target.data).sum().item()

            # test loss average
            adv_loss_avg += float(adv_loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)
    state['adv_test_loss'] = adv_loss_avg / len(test_loader)
    state['adv_test_accuracy'] = adv_correct / len(test_loader.dataset)

def test_in_trainset():
    train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=opt.test_bs, shuffle=False,
    num_workers=opt.prefetch, pin_memory=torch.cuda.is_available())
    net.eval()
    loss_avg = 0.0
    correct = 0
    adv_loss_avg = 0.0
    adv_correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()

            adv_data = adversary_test(net, data, target)

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

            # forward
            adv_output = net(adv_data)
            adv_loss = F.cross_entropy(adv_output, target)

            # accuracy
            adv_pred = adv_output.data.max(1)[1]
            adv_correct += adv_pred.eq(target.data).sum().item()

            # test loss average
            adv_loss_avg += float(adv_loss.data)

    state['train_loss'] = loss_avg / len(train_loader)
    state['train_accuracy'] = correct / len(train_loader.dataset)
    state['adv_train_loss'] = adv_loss_avg / len(train_loader)
    state['adv_train_accuracy'] = adv_correct / len(train_loader.dataset)

opt = PrivateOptions().parse()

state = {k: v for k, v in opt._get_kwargs()}

torch.manual_seed(opt.random_seed)
np.random.seed(opt.random_seed)
cudnn.benchmark = True

# # mean and standard deviation of channels of CIFAR-10 images
# mean = [x / 255 for x in [125.3, 123.0, 113.9]]
# std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(
                                         mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(
                                         mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

if opt.dataset == 'cifar10':
    train_data = dset.CIFAR10(opt.dataroot, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10(opt.dataroot, train=False, transform=test_transform)
    num_classes = 10
else:
    train_data = dset.CIFAR100(opt.dataroot, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100(opt.dataroot, train=False, transform=test_transform)
    num_classes = 100

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=opt.batch_size, shuffle=True,
    num_workers=opt.prefetch, pin_memory=torch.cuda.is_available())
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=opt.test_bs, shuffle=False,
    num_workers=opt.prefetch, pin_memory=torch.cuda.is_available())

# Create model
if opt.model == 'wrn':
    net = WideResNet(opt.layers, num_classes, opt.widen_factor, dropRate=opt.droprate)
else:
    assert False, opt.model + ' is not supported.'

start_epoch = opt.start_epoch

if opt.ngpu > 0:
    net = torch.nn.DataParallel(net, device_ids=list(range(opt.ngpu)))
    net.cuda()
    torch.cuda.manual_seed(opt.random_seed)

# Restore model if desired
if opt.load != '':
    if opt.test and os.path.isfile(opt.load):
        net.load_state_dict(torch.load(opt.load))
        print('Appointed Model Restored!')
    else:
        model_name = os.path.join(opt.load, opt.dataset + opt.model +
                                  '_epoch_' + str(start_epoch) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', start_epoch)
        else:
            raise Exception("Could not resume")

epoch_step = json.loads(opt.epoch_step)
lr = state['learning_rate']
optimizer = torch.optim.SGD(
    net.parameters(), lr, momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


# def cosine_annealing(step, total_steps, lr_max, lr_min):
#     return lr_min + (lr_max - lr_min) * 0.5 * (
#             1 + np.cos(step / total_steps * np.pi))
#
#
# scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer,
#     lr_lambda=lambda step: cosine_annealing(
#         step,
#         opt.epochs * len(train_loader),
#         1,  # since lr_lambda computes multiplicative factor
#         1e-6 / opt.learning_rate))  # originally 1e-6

adversary_train = pgd.PGD(epsilon=opt.epsilon * 2, num_steps=opt.num_steps, step_size=opt.step_size * 2).cuda()
adversary_test = pgd.PGD(epsilon=opt.epsilon * 2, num_steps=opt.test_num_steps, step_size=opt.test_step_size * 2).cuda()

if opt.test:
    test_in_testset()
    # test_in_trainset()
    print(state)
    exit()

# Make save directory
if not os.path.exists(opt.save):
    os.makedirs(opt.save)
if not os.path.isdir(opt.save):
    raise Exception('%s is not a dir' % opt.save)

with open(os.path.join(opt.save, "log_" + opt.dataset + opt.model +
                                  '_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_accuracy(%)\n')

print('Beginning Training\n')

# Main loop
best_test_accuracy = 0
for epoch in range(start_epoch, opt.epochs + 1):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()

    # Save model
    if epoch > 10 and epoch % 10 == 0:
        torch.save(net.state_dict(),
                   os.path.join(opt.save, opt.dataset + opt.model +
                                '_epoch_' + str(epoch) + '.pt'))

    if state['test_accuracy'] > best_test_accuracy:
        best_test_accuracy = state['test_accuracy']
        torch.save(net.state_dict(),
                   os.path.join(opt.save, opt.dataset + opt.model +
                                '_epoch_best.pt'))

    # Show results
    with open(os.path.join(opt.save, "log_" + opt.dataset + opt.model +
                                      '_training_results.csv'), 'a') as f:
        f.write('%03d,%0.6f,%05d,%0.3f,%0.3f,%0.2f\n' % (
            (epoch),
            lr,
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100. * state['test_accuracy'],
        ))

    print('Epoch {0:3d} | LR {1:.6f} | Time {2:5d} | Train Loss {3:.3f} | Test Loss {4:.3f} | Test Acc {5:.2f}'.format(
        (epoch),
        lr,
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100. * state['test_accuracy'])
    )

    # Adjust learning rate
    if epoch in epoch_step:
        lr = optimizer.param_groups[0]['lr'] * opt.lr_decay_ratio
        optimizer = torch.optim.SGD(
            net.parameters(), lr, momentum=state['momentum'],
            weight_decay=state['decay'], nesterov=True)
        print("new lr:", lr)
