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
import cPickle as pickle


class PrivateOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # WRN Architecture
        self.parser.add_argument('--layers', default=28, type=int, help='total number of layers')
        self.parser.add_argument('--widen-factor', default=10, type=int, help='widen factor')
        self.parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')

        self.parser.set_defaults(batch_size=1)
        self.parser.set_defaults(test_bs=1)

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

if opt.ngpu > 0:
    net = torch.nn.DataParallel(net, device_ids=list(range(opt.ngpu)))
    net.cuda()
    torch.cuda.manual_seed(opt.random_seed)

# Restore model if desired
if opt.load != '' and os.path.isfile(opt.load):
    net.load_state_dict(torch.load(opt.load))
    print('Appointed Model Restored!')
else:
    raise Exception("Could not resume")

adversary_test = pgd.PGD(epsilon=opt.epsilon * 2, num_steps=opt.test_num_steps, step_size=opt.test_step_size * 2).cuda()

net.eval()
index = 0
for data, target in test_loader:
    data, target = data.cuda(), target.cuda()

    adv_data = adversary_test(net, data, target)

    # forward
    output = net(adv_data)
    loss = F.cross_entropy(output, target)
    loss.backward()

    g = adv_data.grad
    g = g.data.cpu().numpy()
    g = g.transpose(0, 2, 3, 1)
    g = g.squeeze(axis=0)
    filename = 'boundry/gradients/{}_{}_{}_{}_cifar.pickle'.format('pgd', 'wrn', index, target[0])
    print(filename)
    pickle.dump(g, open(filename, 'w'), pickle.HIGHEST_PROTOCOL)
    index += 1

    break




